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
            r=96, lora_alpha=192, bias="none",
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
    # ================= 🔥 新增的批量推理方法 (终极 Oracle 版: 先验 fov_x + 后处理重构) 🔥 =================
    def infer_batch(self, images: list, intrinsics: list = None):
        import math # 确保使用三角函数
        
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
        
        # 3. 构造 Padding 张量 & FOV 张量
        batched_tensor = torch.zeros((B, 3, max_h, max_w), device=self.device).half()
        batched_fov_x = torch.zeros((B,), device=self.device, dtype=torch.float32)
        has_fov = False
        
        for i, img in enumerate(images):
            h, w = new_shapes[i]
            h_orig, w_orig = orig_shapes[i]
            
            img = img.to(self.device).half().unsqueeze(0)
            if (h, w) != (h_orig, w_orig):
                img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
            batched_tensor[i, :, :h, :w] = img.squeeze(0)
            
            # 🔥 新增：将评测传入的绝对像素内参转化为 fov_x 角度
            if intrinsics is not None and intrinsics[i] is not None:
                intr = intrinsics[i]
                fx = intr[0, 0].item() # 提取 X 轴物理焦距 (像素单位)
                
                # 物理公式: FOV_x = 2 * arctan(W / (2 * fx))
                fov_x_rad = 2 * math.atan(w_orig / (2 * fx))
                batched_fov_x[i] = math.degrees(fov_x_rad)
                has_fov = True

        # 4. 执行推理 (🚨 动态传入 fov_x 字典参数)
        infer_kwargs = {}
        if has_fov:
            infer_kwargs['fov_x'] = batched_fov_x
            
        with torch.inference_mode():
            if hasattr(self.model, 'infer'):
                output = self.model.infer(batched_tensor, **infer_kwargs)
            elif hasattr(self.model.base_model.model, 'infer'):
                output = self.model.base_model.model.infer(batched_tensor, **infer_kwargs)
            else:
                output = self.model(batched_tensor, **infer_kwargs)

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
                
                # Case A: Points
                if v_i.ndim == 3 and v_i.shape[0] == max_h and v_i.shape[1] == max_w and v_i.shape[2] == 3:
                    v_i = v_i[:new_h, :new_w, :] 
                    if needs_resize:
                        v_perm = v_i.permute(2, 0, 1).unsqueeze(0)
                        v_res = F.interpolate(v_perm, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0).permute(1, 2, 0)
                
                # Case B: Normal
                elif v_i.ndim == 3 and v_i.shape[0] == 3 and v_i.shape[1] == max_h and v_i.shape[2] == max_w:
                    v_i = v_i[:, :new_h, :new_w]
                    if needs_resize:
                        v_uns = v_i.unsqueeze(0)
                        v_res = F.interpolate(v_uns, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                        v_i = v_res.squeeze(0)
                
                # Case C: Depth/Mask
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

            # ==========================================
            # 🔥 6. Oracle 后处理：根据高质量预测 Depth 绝对重构真值 Points
            # ==========================================
            if intrinsics is not None and intrinsics[i] is not None and 'depth' in pred_i:
                intr = intrinsics[i].to(self.device).float()
                depth = pred_i['depth'].float()
                
                # 生成原图尺寸的像素坐标网格
                v, u = torch.meshgrid(
                    torch.arange(h_orig, device=self.device, dtype=torch.float32), 
                    torch.arange(w_orig, device=self.device, dtype=torch.float32), 
                    indexing='ij'
                )
                
                fx, fy = intr[0, 0], intr[1, 1]
                cx, cy = intr[0, 2], intr[1, 2]
                
                # 逆透视投影 (Unproject)
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                
                # 覆盖原点云预测，同时写入精确内参
                pred_i['points'] = torch.stack([x, y, z], dim=-1).half() 
                pred_i['intrinsics'] = intr

            # 绑定 Metric 评测用的 Key
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
