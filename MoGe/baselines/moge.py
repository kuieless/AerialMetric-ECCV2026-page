# # # # import os
# # # # import sys
# # # # from typing import *
# # # # import importlib

# # # # import click
# # # # import torch
# # # # import utils3d

# # # # from moge.test.baseline import MGEBaselineInterface


# # # # class Baseline(MGEBaselineInterface):

# # # #     def __init__(self, num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
# # # #         super().__init__()
# # # #         from moge.model import import_model_class_by_version
# # # #         MoGeModel = import_model_class_by_version(version)
# # # #         self.version = version

# # # #         self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        
# # # #         self.device = torch.device(device)
# # # #         self.num_tokens = num_tokens
# # # #         self.resolution_level = resolution_level
# # # #         self.use_fp16 = use_fp16
    
# # # #     @click.command()
# # # #     @click.option('--num_tokens', type=int, default=None)
# # # #     @click.option('--resolution_level', type=int, default=9)
# # # #     @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl')
# # # #     @click.option('--fp16', 'use_fp16', is_flag=True)
# # # #     @click.option('--device', type=str, default='cuda:0')
# # # #     @click.option('--version', type=str, default='v1')
# # # #     @staticmethod
# # # #     def load(num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
# # # #         return Baseline(num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version)

# # # #     # Implementation for inference
# # # #     @torch.inference_mode()
# # # #     def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
# # # #         if intrinsics is not None:
# # # #             fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
# # # #             fov_x = torch.rad2deg(fov_x)
# # # #         else:
# # # #             fov_x = None
# # # #         output = self.model.infer(image, fov_x=fov_x, apply_mask=True, num_tokens=self.num_tokens)
        
# # # #         if self.version == 'v1':
# # # #             return {
# # # #                 'points_scale_invariant': output['points'],
# # # #                 'depth_scale_invariant': output['depth'],
# # # #                 'intrinsics': output['intrinsics'],
# # # #             }
# # # #         else:
# # # #             return {
# # # #                 'points_metric': output['points'],
# # # #                 'depth_metric': output['depth'],
# # # #                 'intrinsics': output['intrinsics'],
# # # #             }

# # # #     @torch.inference_mode()
# # # #     def infer_for_evaluation(self, image: torch.FloatTensor, intrinsics: torch.FloatTensor = None):
# # # #         if intrinsics is not None:
# # # #             fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
# # # #             fov_x = torch.rad2deg(fov_x)
# # # #         else:
# # # #             fov_x = None
# # # #         output = self.model.infer(image, fov_x=fov_x, apply_mask=False, num_tokens=self.num_tokens, use_fp16=self.use_fp16)
        
# # # #         if self.version == 'v1':
# # # #             return {
# # # #                 'points_scale_invariant': output['points'],
# # # #                 'depth_scale_invariant': output['depth'],
# # # #                 'intrinsics': output['intrinsics'],
# # # #             }
# # # #         else:
# # # #             return {
# # # #                 'points_metric': output['points'],
# # # #                 'depth_metric': output['depth'],
# # # #                 'intrinsics': output['intrinsics'],
# # # #             }
        

# # # import sys
# # # from pathlib import Path
# # # import torch
# # # import click  # <--- 必须导入这个

# # # # 引用 v2 模型
# # # from moge.model.v2 import MoGeModel
# # # from moge.test.baseline import MGEBaselineInterface

# # # class Baseline(MGEBaselineInterface):
# # #     def __init__(self, num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version='v2'):
# # #         super().__init__(num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version)
        
# # #         # 加载 v2 模型
# # #         self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        
# # #         if use_fp16:
# # #             self.model = self.model.half()
            
# # #         self.resolution_level = resolution_level

# # #     def infer_for_evaluation(self, image, intrinsics=None):
# # #         # v2 推理接口
# # #         return self.model.infer(
# # #             image, 
# # #             resolution_level=self.resolution_level
# # #         )

# # #     # =========================================================================
# # #     # 核心修复：添加 @click.command 和 @click.option 装饰器
# # #     # 这样 eval_baseline.py 才能调用 .main()
# # #     # =========================================================================
# # #     @classmethod
# # #     @click.command(context_settings={'ignore_unknown_options': True})
# # #     @click.option('--pretrained', type=str, required=True, help='Path to the pretrained model.')
# # #     @click.option('--resolution_level', type=int, default=9, help='Resolution level.')
# # #     @click.option('--use_fp16', is_flag=True, help='Use fp16.')
# # #     def load(cls, pretrained, resolution_level, use_fp16, **kwargs):
# # #         # Click 会解析命令行参数并传给这里，而不是传 args 对象
# # #         return cls(
# # #             num_tokens=None,
# # #             resolution_level=resolution_level,
# # #             pretrained_model_name_or_path=pretrained,
# # #             use_fp16=use_fp16,
# # #             device=torch.device('cuda'),
# # #             version='v2'
# # #         )
# # import sys
# # from pathlib import Path
# # import torch
# # import click

# # # 引用 v2 模型
# # from moge.model.v2 import MoGeModel
# # from moge.test.baseline import MGEBaselineInterface

# # class Baseline(MGEBaselineInterface):
# #     def __init__(self, num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version='v2'):
# #         super().__init__(num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version)
        
# #         # 加载 v2 模型
# #         self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        
# #         if use_fp16:
# #             self.model = self.model.half()
            
# #         self.resolution_level = resolution_level

# #     def infer_for_evaluation(self, image, intrinsics=None):
# #         return self.model.infer(
# #             image, 
# #             resolution_level=self.resolution_level
# #         )

# #     # =========================================================================
# #     # 修复点：去掉 @classmethod，去掉参数中的 cls
# #     # =========================================================================
# #     @click.command(context_settings={'ignore_unknown_options': True})
# #     @click.option('--pretrained', type=str, required=True, help='Path to the pretrained model.')
# #     @click.option('--resolution_level', type=int, default=9, help='Resolution level.')
# #     @click.option('--use_fp16', is_flag=True, help='Use fp16.')
# #     def load(pretrained, resolution_level, use_fp16, **kwargs):
# #         # 直接返回 Baseline 实例，不再依赖 cls 参数
# #         return Baseline(
# #             num_tokens=None,
# #             resolution_level=resolution_level,
# #             pretrained_model_name_or_path=pretrained,
# #             use_fp16=use_fp16,
# #             device=torch.device('cuda'),
# #             version='v2'
# #         )
# import sys
# from pathlib import Path
# import torch
# import click

# # 引用 v2 模型 (确保你之前已经保存了 moge/model/v2.py)
# from moge.model.v2 import MoGeModel
# from moge.test.baseline import MGEBaselineInterface

# class Baseline(MGEBaselineInterface):
#     def __init__(self, num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version='v2'):
#         # =========================================================
#         # 修复点：父类不需要参数，直接空的初始化即可
#         # =========================================================
#         super().__init__() 
        
#         # 加载 v2 模型
#         self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        
#         if use_fp16:
#             self.model = self.model.half()
            
#         self.resolution_level = resolution_level

#     def infer_for_evaluation(self, image, intrinsics=None):
#         # 推理逻辑
#         return self.model.infer(
#             image, 
#             resolution_level=self.resolution_level
#         )

#     # 这里的装饰器必须保留，因为 eval_baseline.py 需要它来解析参数
#     @click.command(context_settings={'ignore_unknown_options': True})
#     @click.option('--pretrained', type=str, required=True, help='Path to the pretrained model.')
#     @click.option('--resolution_level', type=int, default=9, help='Resolution level.')
#     @click.option('--use_fp16', is_flag=True, help='Use fp16.')
#     def load(pretrained, resolution_level, use_fp16, **kwargs):
#         # 返回实例化对象
#         return Baseline(
#             num_tokens=None,
#             resolution_level=resolution_level,
#             pretrained_model_name_or_path=pretrained,
#             use_fp16=use_fp16,
#             device=torch.device('cuda'),
#             version='v2'
#         )

# import sys
# from pathlib import Path
# import torch
# import click

# # 引用 v2 模型
# from moge.model.v2 import MoGeModel
# from moge.test.baseline import MGEBaselineInterface

# class Baseline(MGEBaselineInterface):
#     def __init__(self, num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version='v2'):
#         # 父类初始化
#         super().__init__() 
        
#         # =========================================================
#         # 🛠️ 修复点：显式保存 device 属性，防止报错
#         # =========================================================
#         self.device = device 
#         # =========================================================

#         # 加载 v2 模型
#         self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(self.device).eval()
        
#         if use_fp16:
#             self.model = self.model.half()
            
#         self.resolution_level = resolution_level

#     def infer_for_evaluation(self, image, intrinsics=None):
#         output = self.model.infer(
#             image, 
#             resolution_level=self.resolution_level
#         )
#         return {
#             'points_metric': output['points'],  # 关键！
#             'depth_metric': output['depth'],    # 关键！
#             'intrinsics': output['intrinsics'],
#             'mask': output.get('mask', None)
#         }

#     @click.command(context_settings={'ignore_unknown_options': True})
#     @click.option('--pretrained', type=str, required=True, help='Path to the pretrained model.')
#     @click.option('--resolution_level', type=int, default=9, help='Resolution level.')
#     @click.option('--use_fp16', is_flag=True, help='Use fp16.')
#     def load(pretrained, resolution_level, use_fp16, **kwargs):
#         return Baseline(
#             num_tokens=None,
#             resolution_level=resolution_level,
#             pretrained_model_name_or_path=pretrained,
#             use_fp16=use_fp16,
#             device=torch.device('cuda'), # 这里定义了设备
#             version='v2'
#         )
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import click

# 引用 v2 模型
from moge.model.v2 import MoGeModel
from moge.test.baseline import MGEBaselineInterface

class Baseline(MGEBaselineInterface):
    def __init__(self, num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version='v2'):
        super().__init__() 
        self.device = device 
        
        # 加载 v2 模型
        self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(self.device).eval()
        
        if use_fp16:
            self.model = self.model.half()
            
        self.resolution_level = resolution_level

    # ================= 🔥 对齐 LoRA 脚本的批量推理核心 🔥 =================
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
        
        # 3. 构造 Padding 张量 (动态匹配 fp16/fp32)
        dtype = next(self.model.parameters()).dtype
        batched_tensor = torch.zeros((B, 3, max_h, max_w), device=self.device, dtype=dtype)
        
        for i, img in enumerate(images):
            h, w = new_shapes[i]
            img = img.to(self.device, dtype=dtype).unsqueeze(0)
            if (h, w) != orig_shapes[i]:
                img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
            batched_tensor[i, :, :h, :w] = img.squeeze(0)

        # 4. 执行推理 (不再传 resolution_level，走底层前向传播)
        with torch.inference_mode():
            output = self.model.infer(batched_tensor)

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
                
                v_i = v[i] 
                while v_i.dim() > 1 and v_i.shape[0] == 1:
                    v_i = v_i.squeeze(0)
                
                # Case A: Points (max_h, max_w, 3)
                if v_i.ndim == 3 and v_i.shape[0] == max_h and v_i.shape[1] == max_w and v_i.shape[2] == 3:
                    v_i = v_i[:new_h, :new_w, :]
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
                        v_res = F.normalize(v_res, p=2, dim=1)  # 🛠️ 修复：法线归一化
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
    
    def infer_for_evaluation(self, image, intrinsics=None):
        return self.infer_batch([image])[0]

    @click.command(context_settings={'ignore_unknown_options': True})
    @click.option('--pretrained', type=str, required=True, help='Path to the pretrained model.')
    @click.option('--resolution_level', type=int, default=9, help='Resolution level.')
    @click.option('--use_fp16', is_flag=True, help='Use fp16.')
    def load(pretrained, resolution_level, use_fp16, **kwargs):
        return Baseline(
            num_tokens=None,
            resolution_level=resolution_level,
            pretrained_model_name_or_path=pretrained,
            use_fp16=use_fp16,
            device=torch.device('cuda'),
            version='v2'
        )