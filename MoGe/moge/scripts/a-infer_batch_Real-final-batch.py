import argparse
import os
import sys
import json
import time
import logging
import traceback
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib

# 设置无头模式，防止服务器报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 🔧 0. 参数配置 =================

def setup_args():
    parser = argparse.ArgumentParser(description="MoGe Inference Script (Parallel)")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--input_root", type=str, required=True, help="数据集输入根目录")
    parser.add_argument("--output_root", type=str, required=True, help="结果输出根目录")
    parser.add_argument("--project_root", type=str, default="/home/szq/moge2/MoGe", help="MoGe项目根目录")
    
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    parser.add_argument("--resize", type=int, default=None, help="缩放长边大小")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖")
    
    # 🔥🔥🔥 并行分片参数 🔥🔥🔥
    parser.add_argument("--total_shards", type=int, default=1, help="并行总进程数")
    parser.add_argument("--shard_id", type=int, default=0, help="当前进程ID (0 ~ total-1)")
    
    return parser.parse_args()

# ================= 🧠 1. 推理引擎 (完全照搬你的成功脚本) =================

class MoGeInferenceEngine:
    def __init__(self, model_path, version="v2", device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        
        # 只在第一个进程打印，保持清爽
        if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
            print(f"\n📦 [系统] 正在加载模型权重...")
            print(f"   path: {model_path}")

        try:
            from moge.model import import_model_class_by_version
            ModelClass = import_model_class_by_version(version)
            
            # 🔥 核心：直接使用官方提供的加载方法，不做额外修改
            # 这里的 model_path 必须是干净的字符串
            self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
            
            if self.fp16:
                self.model.half()
                
            if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
                print("✅ [系统] 模型加载完成！准备就绪。\n")
                
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            # 如果这里报错 Repo id，说明 model_path 路径对应的文件没找到，或者不是文件
            print(f"🔍 请检查路径是否存在: {os.path.exists(model_path)}")
            raise e

    def process_scene(self, input_dir, output_dir, config):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in VALID_IMG_EXTS])

        if not image_files: return

        # 推理循环
        for img_path in image_files:
            rel_path = img_path.relative_to(input_path)
            save_dir = output_path / rel_path.parent / img_path.stem
            
            if not config.get('overwrite', False):
                if (save_dir / 'depth.npy').exists():
                    continue

            try:
                self._infer_single_image(img_path, save_dir, config)
            except torch.cuda.OutOfMemoryError:
                print(f"💥 [OOM] Skip: {img_path.name}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ [Error] {img_path.name}: {e}")

    def _infer_single_image(self, img_path, save_dir, config):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None: return

        h, w = image_bgr.shape[:2]
        process_bgr = image_bgr
        resize_to = config.get('resize')
        
        # 缩放逻辑
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

        with torch.inference_mode():
            output = self.model.infer(image_tensor, resolution_level=9, use_fp16=self.fp16)

        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr)
        np.save(str(save_dir / 'depth.npy'), output['depth'].cpu().numpy())
        

        if 'metric_scale' in output:
            with open(save_dir / 'scale.json', 'w') as f:
                json.dump({'metric_scale': output['metric_scale'].float().cpu().item()}, f, indent=4)

        if 'intrinsics' in output:
            import utils3d
            intrinsics = output['intrinsics'].cpu().numpy()
            fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
            with open(save_dir / 'fov.json', 'w') as f:
                json.dump({'fov_x': float(np.rad2deg(fov_x)), 'fov_y': float(np.rad2deg(fov_y))}, f, indent=4)

        if config.get('save_maps', True):
            self._save_visualization(output['depth'].cpu().numpy(), save_dir / 'depth_vis.png')

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

# ================= 🚀 2. 主程序 (集成并行逻辑) =================

def main():
    args = setup_args()
    
    # 注入全局变量
    global shard_id
    shard_id = args.shard_id

    # 🔥 重要：在 import moge 之前设置 path
    if args.project_root not in sys.path:
        sys.path.insert(0, args.project_root)
        print(f"🐍 [Env] Added project root: {args.project_root}")
    
    # 简单的路径清洗，防止 shell 脚本传参带换行符导致报错
    clean_model_path = args.model_path.strip()
    
    if not os.path.isfile(clean_model_path):
        print(f"❌ [Error] 模型文件不存在: '{clean_model_path}'")
        # 只有在路径不对时，from_pretrained 才会去连 huggingface 报错
        sys.exit(1)

    # 1. 初始化引擎
    # 直接使用你脚本里验证过的逻辑
    config = {"resize": args.resize, "overwrite": args.overwrite, "save_maps": True}
    engine = MoGeInferenceEngine(clean_model_path, version="v2", device=args.device, fp16=True)

    # 2. 扫描任务
    if args.shard_id == 0:
        print(f"📂 Scanning input root: {args.input_root}")
        
    CANDIDATE_IMG_DIRS = ["rgbs", "images_downsampled", "images", "img", "rgb", "image"]
    VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png')
    tasks = []

    if os.path.exists(args.input_root):
        scene_names = sorted(os.listdir(args.input_root))
        for scene_name in scene_names:
            scene_path = os.path.join(args.input_root, scene_name)
            if not os.path.isdir(scene_path): continue
            
            input_rgbs_path = None
            for candidate in CANDIDATE_IMG_DIRS:
                p = os.path.join(scene_path, candidate)
                if os.path.isdir(p) and len(os.listdir(p)) > 0:
                    input_rgbs_path = p
                    break
            
            if not input_rgbs_path:
                 if any(f.endswith(VALID_IMG_EXTS) for f in os.listdir(scene_path)):
                     input_rgbs_path = scene_path

            if input_rgbs_path:
                tasks.append({"input": input_rgbs_path, "output": os.path.join(args.output_root, scene_name)})

    if args.shard_id == 0:
        print(f"✅ 总任务数: {len(tasks)} | 并行分片: {args.total_shards}")

    # 3. 执行任务 (并行分片)
    my_tasks_count = 0
    for i, task in enumerate(tasks):
        # 核心：取模运算分配任务
        if i % args.total_shards != args.shard_id:
            continue
            
        print(f"[Worker {args.shard_id}] 处理: {os.path.basename(task['output'])}")
        engine.process_scene(task['input'], task['output'], config)
        my_tasks_count += 1
        
        if my_tasks_count % 10 == 0: torch.cuda.empty_cache()

    print(f"🎉 [Worker {args.shard_id}] 完成！共处理 {my_tasks_count} 个场景。")

if __name__ == "__main__":
    main()