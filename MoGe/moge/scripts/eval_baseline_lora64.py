import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import click
from peft import LoraConfig, get_peft_model

# 尝试导入 moge 相关模块
try:
    from moge.test.baseline import MGEBaselineInterface
    from moge.model import import_model_class_by_version
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from moge.test.baseline import MGEBaselineInterface
    from moge.model import import_model_class_by_version

class Baseline(MGEBaselineInterface):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    @click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
    @click.option('--lora_config', required=True, help='Path to lora config.json')
    @click.option('--lora_weight', required=True, help='Path to lora .pt file')
    def load(lora_config, lora_weight, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n📦 [LoRA Eval Wrapper] Loading...")
        
        with open(lora_config, 'r') as f:
            train_config = json.load(f)

        model_version = train_config.get('model_version', 'v2')
        MoGeModel = import_model_class_by_version(model_version)
        model = MoGeModel(**train_config['model'])

        LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
        HEADS_TO_SAVE = ["scale_head"]
        peft_config = LoraConfig(
            r=64, lora_alpha=128, bias="none",
            target_modules=LORA_TARGETS, modules_to_save=HEADS_TO_SAVE
        )
        model = get_peft_model(model, peft_config)

        print(f"   Weight: {lora_weight}")
        checkpoint = torch.load(lora_weight, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        new_state_dict = {}
        model_keys = set(model.state_dict().keys())

        for k, v in state_dict.items():
            if k in model_keys:
                new_state_dict[k] = v; continue
            prefixed_k = f"base_model.model.{k}"
            if prefixed_k in model_keys:
                new_state_dict[prefixed_k] = v; continue
            parts = prefixed_k.split('.')
            if parts[-1] in ['weight', 'bias']:
                base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                if base_injected_k in model_keys:
                    new_state_dict[base_injected_k] = v; continue
            for head in HEADS_TO_SAVE:
                if k.startswith(head):
                    suffix = k[len(head)+1:]
                    trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
                    if trainable_k in model_keys:
                        new_state_dict[trainable_k] = v; break

        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ LoRA Loaded. Ready for inference.")

        model.to(device)
        model.eval()
        model.half()

        return Baseline(model, device)

    # ================= 🔥 新增的批量推理方法 🔥 =================
    def infer_batch(self, images: list):
        B = len(images)
        orig_shapes = [(img.shape[1], img.shape[2]) for img in images]
        
        # 1. 计算对齐 14 后的尺寸
        new_shapes = []
        for h, w in orig_shapes:
            new_h, new_w = (h // 14) * 14, (w // 14) * 14
            new_shapes.append((new_h, new_w))
            
        # 2. 获取 Batch 内最大尺寸用于 Padding
        max_h = max(14, max(s[0] for s in new_shapes))
        max_w = max(14, max(s[1] for s in new_shapes))
        
        # 3. 构造 Padding 张量
        batched_tensor = torch.zeros((B, 3, max_h, max_w), device=self.device).half()
        
        for i, img in enumerate(images):
            h, w = new_shapes[i]
            img = img.to(self.device).half().unsqueeze(0)
            if (h, w) != orig_shapes[i]:
                img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
            batched_tensor[i, :, :h, :w] = img.squeeze(0)

        # 4. 执行推理
        with torch.inference_mode():
            if hasattr(self.model, 'infer'):
                output = self.model.infer(batched_tensor)
            elif hasattr(self.model.base_model.model, 'infer'):
                output = self.model.base_model.model.infer(batched_tensor)
            else:
                output = self.model(batched_tensor)

        # 5. 后处理：逐张裁切 Padding 还原原尺寸
        preds = []
        for i in range(B):
            h_orig, w_orig = orig_shapes[i]
            new_h, new_w = new_shapes[i]
            needs_resize = (new_h != h_orig) or (new_w != w_orig)
            
            pred_i = {}
            for k, v in output.items():
                if not isinstance(v, torch.Tensor):
                    pred_i[k] = v
                    continue
                
                v_i = v[i] # 提取第 i 个结果
                while v_i.dim() > 1 and v_i.shape[0] == 1:
                    v_i = v_i.squeeze(0)
                
                # Case A: Points (max_h, max_w, 3)
                if v_i.ndim == 3 and v_i.shape[0] == max_h and v_i.shape[1] == max_w and v_i.shape[2] == 3:
                    v_i = v_i[:new_h, :new_w, :] # 裁掉黑边
                    if needs_resize:
                        v_perm = v_i.permute(2, 0, 1).unsqueeze(0)
                        v_res = F.interpolate(v_perm, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0).permute(1, 2, 0)
                
                # Case B: Normal (3, max_h, max_w)
                elif v_i.ndim == 3 and v_i.shape[0] == 3 and v_i.shape[1] == max_h and v_i.shape[2] == max_w:
                    v_i = v_i[:, :new_h, :new_w]
                    if needs_resize:
                        v_uns = v_i.unsqueeze(0)
                        v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0)
                
                # Case C: Depth/Mask (max_h, max_w)
                elif v_i.ndim == 2 and v_i.shape[0] == max_h and v_i.shape[1] == max_w:
                    v_i = v_i[:new_h, :new_w]
                    if needs_resize:
                        is_bool = (v_i.dtype == torch.bool)
                        if is_bool: v_i = v_i.float()
                        v_uns = v_i.unsqueeze(0).unsqueeze(0)
                        v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0).squeeze(0)
                        if is_bool or k == 'mask':
                            v_i = (v_i > 0.5)

                pred_i[k] = v_i

            # Metric Keys
            if 'depth' in pred_i: pred_i['depth_metric'] = pred_i['depth']
            if 'points' in pred_i: pred_i['points_metric'] = pred_i['points']
            preds.append(pred_i)

        return preds

    def infer_batch_for_evaluation(self, images: list, intrinsics=None):
        return self.infer_batch(images)
    
    # 兼容原有的单张推理调用
    def infer(self, image: torch.Tensor):
        return self.infer_batch([image])[0]
    
    def infer_for_evaluation(self, image, intrinsics=None):
        return self.infer(image)
# import sys
# import json
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# import click
# from peft import LoraConfig, get_peft_model

# # 尝试导入 moge 相关模块
# try:
#     from moge.test.baseline import MGEBaselineInterface
#     from moge.model import import_model_class_by_version
# except ImportError:
#     sys.path.append(str(Path(__file__).resolve().parents[2]))
#     from moge.test.baseline import MGEBaselineInterface
#     from moge.model import import_model_class_by_version

# class Baseline(MGEBaselineInterface):
#     def __init__(self, model, device):
#         super().__init__()
#         self.model = model
#         self.device = device

#     @click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
#     @click.option('--lora_config', required=True, help='Path to lora config.json')
#     @click.option('--lora_weight', required=True, help='Path to lora .pt file')
#     def load(lora_config, lora_weight, **kwargs):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"\n📦 [LoRA Eval Wrapper] Loading...")
        
#         with open(lora_config, 'r') as f:
#             train_config = json.load(f)

#         model_version = train_config.get('model_version', 'v2')
#         MoGeModel = import_model_class_by_version(model_version)
#         model = MoGeModel(**train_config['model'])

#         LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
#         HEADS_TO_SAVE = ["scale_head"]
#         peft_config = LoraConfig(
#             r=32, lora_alpha=64, bias="none",
#             target_modules=LORA_TARGETS, modules_to_save=HEADS_TO_SAVE
#         )
#         model = get_peft_model(model, peft_config)

#         print(f"   Weight: {lora_weight}")
#         checkpoint = torch.load(lora_weight, map_location='cpu')
#         state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

#         new_state_dict = {}
#         model_keys = set(model.state_dict().keys())

#         for k, v in state_dict.items():
#             if k in model_keys:
#                 new_state_dict[k] = v; continue
#             prefixed_k = f"base_model.model.{k}"
#             if prefixed_k in model_keys:
#                 new_state_dict[prefixed_k] = v; continue
#             parts = prefixed_k.split('.')
#             if parts[-1] in ['weight', 'bias']:
#                 base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
#                 if base_injected_k in model_keys:
#                     new_state_dict[base_injected_k] = v; continue
#             for head in HEADS_TO_SAVE:
#                 if k.startswith(head):
#                     suffix = k[len(head)+1:]
#                     trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
#                     if trainable_k in model_keys:
#                         new_state_dict[trainable_k] = v; break

#         model.load_state_dict(new_state_dict, strict=False)
#         print(f"✅ LoRA Loaded. Ready for inference.")

#         model.to(device)
#         model.eval()
#         model.half()

#         return Baseline(model, device)

#     def infer(self, image: torch.Tensor):
#         # image input: (3, H_orig, W_orig)
#         h_orig, w_orig = image.shape[1], image.shape[2]

#         # 增加 Batch 维度: (1, 3, H, W)
#         image = image.unsqueeze(0).to(self.device).half()
        
#         # ViT 对齐 14
#         new_h = (h_orig // 14) * 14
#         new_w = (w_orig // 14) * 14
#         needs_resize = (new_h != h_orig) or (new_w != w_orig)

#         if needs_resize:
#             image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

#         with torch.inference_mode():
#             if hasattr(self.model, 'infer'):
#                 output = self.model.infer(image)
#             elif hasattr(self.model.base_model.model, 'infer'):
#                 output = self.model.base_model.model.infer(image)
#             else:
#                 output = self.model(image)

#         # === 后处理逻辑 ===
#         final_output = {}
        
#         # 调试打印 (只打印一次)
#         if not hasattr(self, '_has_printed_debug'):
#             print(f"\n🔍 [Debug Info] Processing first image...")
#             print(f"   Input Size: ({h_orig}, {w_orig}) -> Model Input: ({new_h}, {new_w})")
#             self._has_printed_debug = True

#         for k, v in output.items():
#             if not isinstance(v, torch.Tensor):
#                 final_output[k] = v
#                 continue
            
#             # 1. 移除 Batch 维度 (Squeeze until valid)
#             # 只要 dim 0 是 1 且总维度 > 1 (避免把标量压没了)，就移除
#             while v.dim() > 1 and v.shape[0] == 1:
#                 v = v.squeeze(0)

#             # 2. 如果不需要 Resize，直接使用
#             if not needs_resize:
#                 result = v
#             else:
#                 # 3. 智能 Resize
#                 is_bool = (v.dtype == torch.bool)
#                 if is_bool: v = v.float()
                
#                 result = v # Default

#                 # Case A: Points (H_pad, W_pad, 3) -> 最容易导致 IndexError 的情况
#                 if v.ndim == 3 and v.shape[0] == new_h and v.shape[1] == new_w and v.shape[2] == 3:
#                     v_perm = v.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
#                     v_res = F.interpolate(v_perm, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
#                     result = v_res.squeeze(0).permute(1, 2, 0) # (H, W, 3)

#                 # Case B: Normal (3, H_pad, W_pad)
#                 elif v.ndim == 3 and v.shape[0] == 3 and v.shape[1] == new_h and v.shape[2] == new_w:
#                     v_uns = v.unsqueeze(0)
#                     v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
#                     result = v_res.squeeze(0)

#                 # Case C: Depth/Mask (H_pad, W_pad)
#                 elif v.ndim == 2 and v.shape[0] == new_h and v.shape[1] == new_w:
#                     v_uns = v.unsqueeze(0).unsqueeze(0)
#                     v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
#                     result = v_res.squeeze(0).squeeze(0)

#                 if is_bool or k == 'mask':
#                     result = (result > 0.5)
            
#             final_output[k] = result

#         # === 🔥 关键新增：Key 映射 🔥 ===
#         # metrics.py 期望看到 'depth_metric' 和 'points_metric'
#         # 如果我们不给它，它就不算！
#         if 'depth' in final_output:
#             final_output['depth_metric'] = final_output['depth']
        
#         if 'points' in final_output:
#             final_output['points_metric'] = final_output['points']

#         return final_output

#     def infer_for_evaluation(self, image, intrinsics=None):
#         return self.infer(image)