# import argparse
# import os
# import sys
# import json
# import time
# import logging
# import traceback
# import numpy as np
# import torch
# import cv2
# from pathlib import Path
# from tqdm import tqdm
# import matplotlib

# # 设置无头模式，防止服务器报错
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # ================= 🔧 0. 参数配置 =================

# def setup_args():
#     parser = argparse.ArgumentParser(description="Step 1: Inference (Image-based Sharding)")
#     parser.add_argument("--model_path", type=str, required=True, help="模型权重路径 (.pt)")
#     parser.add_argument("--input_root", type=str, required=True, help="Step 0 输出的 images 目录")
#     parser.add_argument("--output_root", type=str, required=True, help="结果输出根目录")
#     parser.add_argument("--project_root", type=str, default="/home/szq/moge2/MoGe", help="MoGe项目根目录")
    
#     parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
#     parser.add_argument("--resize", type=int, default=None, help="缩放长边大小 (建议 None 或 1024)")
#     parser.add_argument("--overwrite", action="store_true", help="是否覆盖")
    
#     # 🔥🔥🔥 并行分片参数 🔥🔥🔥
#     parser.add_argument("--total_shards", type=int, default=1, help="并行总进程数")
#     parser.add_argument("--shard_id", type=int, default=0, help="当前进程ID (0 ~ total-1)")
    
#     return parser.parse_args()

# # ================= 🧠 1. 推理引擎 =================

# class MoGeInferenceEngine:
#     def __init__(self, model_path, version="v2", device="cuda", fp16=False):
#         self.device = torch.device(device)
#         self.fp16 = fp16
        
#         # 只在第一个进程打印，保持清爽
#         if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
#             print(f"\n📦 [系统] 正在加载模型权重...")
#             print(f"   path: {model_path}")

#         try:
#             from moge.model import import_model_class_by_version
#             ModelClass = import_model_class_by_version(version)
            
#             # 🔥 核心：直接使用官方提供的加载方法
#             self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
            
#             if self.fp16:
#                 self.model.half()
                
#             if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
#                 print("✅ [系统] 模型加载完成！准备就绪。\n")
                
#         except Exception as e:
#             print(f"❌ 模型初始化失败: {e}")
#             print(f"🔍 请检查路径是否存在: {os.path.exists(model_path)}")
#             raise e

#     # 公开的推理接口
#     def infer_single_image(self, img_path, save_dir, config):
#         # 1. 跳过检查
#         if not config.get('overwrite', False):
#             if (save_dir / 'depth.npy').exists():
#                 return

#         # 2. 读取图片
#         image_bgr = cv2.imread(str(img_path))
#         if image_bgr is None: return

#         h, w = image_bgr.shape[:2]
#         process_bgr = image_bgr
#         resize_to = config.get('resize')
        
#         # 3. 缩放逻辑 (限制最大分辨率以防 OOM)
#         if resize_to is not None:
#             scale = resize_to / max(h, w)
#             if scale < 1.0:
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 # 确保是 14 的倍数 (ViT patch size)
#                 new_w = (new_w // 14) * 14
#                 new_h = (new_h // 14) * 14
#                 process_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

#         # 4. 转 Tensor
#         image_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
#         image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).to(self.device)
#         if self.fp16: image_tensor = image_tensor.half()

#         # 5. 推理
#         try:
#             with torch.inference_mode():
#                 output = self.model.infer(image_tensor, resolution_level=9, use_fp16=self.fp16)
#         except torch.cuda.OutOfMemoryError:
#             print(f"💥 [OOM] Skip: {img_path.name}")
#             torch.cuda.empty_cache()
#             return
#         except Exception as e:
#             print(f"❌ [Error] {img_path.name}: {e}")
#             return

#         # 6. 保存结果
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         # 保存原图 (可选，用于对照)
#         # cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr)
        
#         # 保存 NPY
#         np.save(str(save_dir / 'depth.npy'), output['depth'].cpu().numpy())

#         # 保存可视化
#         if config.get('save_maps', True):
#             self._save_visualization(output['depth'].cpu().numpy(), save_dir / 'depth_vis.png')

#     def _save_visualization(self, depth, save_path):
#         valid_mask = (depth > 0) & np.isfinite(depth)
#         if valid_mask.sum() == 0: return
        
#         # 简单的百分比截断，让可视化更好看
#         vmin, vmax = np.percentile(depth[valid_mask], 2), np.percentile(depth[valid_mask], 98)
        
#         plt.figure(figsize=(8, 6))
#         plt.imshow(depth, cmap='Spectral', vmin=vmin, vmax=vmax)
#         plt.axis('off')
#         plt.tight_layout()
#         plt.savefig(str(save_path), dpi=80, bbox_inches='tight')
#         plt.close()

# # ================= 🚀 2. 主程序 (修改为按图片分片) =================

# def main():
#     args = setup_args()
    
#     # 注入全局变量
#     global shard_id
#     shard_id = args.shard_id

#     # 🔥 设置环境变量 path
#     if args.project_root not in sys.path:
#         sys.path.insert(0, args.project_root)
#         if args.shard_id == 0:
#             print(f"🐍 [Env] Added project root: {args.project_root}")
    
#     clean_model_path = args.model_path.strip()
#     if not os.path.isfile(clean_model_path):
#         print(f"❌ [Error] 模型文件不存在: '{clean_model_path}'")
#         sys.exit(1)

#     # 1. 扫描所有图片 (而不是扫描文件夹)
#     # 因为 Step 0 生成的是 mixed_val 文件夹，里面可能有几千张图
#     # 如果按文件夹分片，会导致负载不均衡
#     input_path = Path(args.input_root)
#     if args.shard_id == 0:
#         print(f"📂 Scanning images in: {input_path}")
    
#     VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
#     # rglob('*') 会递归查找 mixed_val 下的所有图片
#     all_images = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in VALID_IMG_EXTS])

#     # 2. 计算当前分片需要处理的图片列表
#     # 列表切片：从 shard_id 开始，每隔 total_shards 取一个
#     my_images = all_images[args.shard_id::args.total_shards]
    
#     if args.shard_id == 0:
#         print(f"✅ 总图片数: {len(all_images)} | Worker 0 分配到: {len(my_images)}")

#     if len(my_images) == 0:
#         print(f"⚠️ [Worker {args.shard_id}] 没有分配到任务，退出。")
#         return

#     # 3. 初始化引擎
#     config = {"resize": args.resize, "overwrite": args.overwrite, "save_maps": True}
#     engine = MoGeInferenceEngine(clean_model_path, version="v2", device=args.device, fp16=True)

#     # 4. 执行循环
#     count = 0
#     pbar = tqdm(my_images, desc=f"Worker {args.shard_id}", position=args.shard_id, leave=False)
    
#     for img_path in pbar:
#         # 计算输出路径:
#         # Input:  .../staging/images/mixed_val/dataset_name_001.jpg
#         # Rel:    mixed_val/dataset_name_001.jpg
#         # Output: .../output_root/mixed_val/dataset_name_001/depth.npy
#         rel_path = img_path.relative_to(input_path)
#         save_dir = Path(args.output_root) / rel_path.parent / img_path.stem
        
#         engine.infer_single_image(img_path, save_dir, config)
        
#         count += 1
#         # 定期清理显存
#         if count % 50 == 0:
#             torch.cuda.empty_cache()

#     print(f"🎉 [Worker {args.shard_id}] 完成！共处理 {count} 张图片。")

# if __name__ == "__main__":
#     main()

import argparse
import os
import sys
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_args():
    parser = argparse.ArgumentParser(description="Step 1: Inference (Mirroring Single Script)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--project_root", type=str, default="/home/szq/moge2/MoGe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser.parse_args()

class MoGeInferenceEngine:
    def __init__(self, model_path, version="v2", device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        
        if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
            print(f"\n📦 [系统] 加载模型: {os.path.basename(model_path)}")

        try:
            from moge.model import import_model_class_by_version
            ModelClass = import_model_class_by_version(version)
            self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
            if self.fp16: self.model.half()
            
            if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
                print("✅ [系统] 模型加载完成。\n")
                
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            sys.exit(1)

    def infer_single_image(self, img_path, save_dir, config):
        if not config.get('overwrite', False):
            if (save_dir / 'depth.npy').exists(): return

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None: return

        h, w = image_bgr.shape[:2]
        process_bgr = image_bgr
        resize_to = config.get('resize')
        
        # 缩放逻辑 (与单机脚本完全一致)
        if resize_to is not None:
            scale = resize_to / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                new_w = (new_w // 14) * 14
                new_h = (new_h // 14) * 14
                process_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).to(self.device)
        if self.fp16: image_tensor = image_tensor.half()

        try:
            with torch.inference_mode():
                # 🔥 与单机脚本一致：infer -> output
                output = self.model.infer(image_tensor, resolution_level=9, use_fp16=self.fp16)
                
                # 🔥 与单机脚本一致：直接取 depth，不做任何额外乘法
                depth = output['depth']

        except torch.cuda.OutOfMemoryError:
            print(f"💥 [OOM] Skip: {img_path.name}")
            torch.cuda.empty_cache()
            return
        except Exception as e:
            print(f"❌ [Error] {img_path.name}: {e}")
            return

        # 保存结果
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存原图
        # cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr) # 为了省空间先注释掉
        
        # 2. 保存 Depth (numpy)
        np.save(str(save_dir / 'depth.npy'), depth.cpu().numpy())
        
        # 3. 保存 Scale (如果有)
        metric_scale = 1.0
        if 'metric_scale' in output:
            metric_scale = output['metric_scale'].float().cpu().item()
        
        with open(save_dir / 'scale.json', 'w') as f:
            json.dump({'metric_scale': metric_scale}, f, indent=4)

        # 4. 保存 FOV (如果有)
        if 'intrinsics' in output:
            import utils3d
            intrinsics = output['intrinsics'].cpu().numpy()
            fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
            with open(save_dir / 'fov.json', 'w') as f:
                json.dump({'fov_x': float(np.rad2deg(fov_x)), 'fov_y': float(np.rad2deg(fov_y))}, f, indent=4)

        # 5. 可视化
        if config.get('save_maps', True):
            self._save_visualization(depth.cpu().numpy(), save_dir / 'depth_vis.png')

    def _save_visualization(self, depth, save_path):
        valid_mask = (depth > 0) & np.isfinite(depth)
        if valid_mask.sum() == 0: return
        vmin, vmax = np.percentile(depth[valid_mask], 2), np.percentile(depth[valid_mask], 98)
        plt.figure(figsize=(8, 6))
        plt.imshow(depth, cmap='Spectral', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=80, bbox_inches='tight')
        plt.close()

def main():
    args = setup_args()
    global shard_id
    shard_id = args.shard_id

    if args.project_root not in sys.path:
        sys.path.insert(0, args.project_root)
        if args.shard_id == 0: print(f"🐍 [Env] Added project root: {args.project_root}")
    
    clean_model_path = args.model_path.strip()
    if not os.path.isfile(clean_model_path):
        print(f"❌ [Error] 模型文件不存在: '{clean_model_path}'")
        sys.exit(1)

    input_path = Path(args.input_root)
    if args.shard_id == 0: print(f"📂 Scanning images in: {input_path}")
    
    VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in VALID_IMG_EXTS])
    my_images = all_images[args.shard_id::args.total_shards]
    
    if args.shard_id == 0: print(f"✅ 总图片数: {len(all_images)} | Worker 0 分配: {len(my_images)}")
    if len(my_images) == 0: return

    # 配置与单机脚本完全一致
    config = {"resize": args.resize, "resolution_level": 9, "overwrite": args.overwrite, "save_maps": True}
    engine = MoGeInferenceEngine(clean_model_path, version="v2", device=args.device, fp16=True)

    count = 0
    pbar = tqdm(my_images, desc=f"Worker {args.shard_id}", position=args.shard_id, leave=False)
    
    for img_path in pbar:
        rel_path = img_path.relative_to(input_path)
        save_dir = Path(args.output_root) / rel_path.parent / img_path.stem
        engine.infer_single_image(img_path, save_dir, config)
        count += 1
        if count % 50 == 0: torch.cuda.empty_cache()

    print(f"🎉 [Worker {args.shard_id}] 完成！共处理 {count} 张图片。")

if __name__ == "__main__":
    main()