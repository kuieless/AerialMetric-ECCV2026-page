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

# 设置无头模式
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 🔧 配置与日志 =================

logging.basicConfig(
    filename='inference_error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_args():
    parser = argparse.ArgumentParser(description="MoGe Inference Script (Parallel)")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--input_root", type=str, required=True, help="数据集输入根目录")
    parser.add_argument("--output_root", type=str, required=True, help="结果输出根目录")
    parser.add_argument("--project_root", type=str, default="/home/szq/moge2/MoGe", help="MoGe项目根目录")
    
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    parser.add_argument("--resize", type=int, default=None, help="缩放长边大小")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖")
    
    # 🔥🔥🔥 [新增] 并行分片参数 🔥🔥🔥
    parser.add_argument("--total_shards", type=int, default=1, help="并行总进程数")
    parser.add_argument("--shard_id", type=int, default=0, help="当前进程ID (0 ~ total-1)")
    
    return parser.parse_args()

# ================= 🧠 推理引擎 =================

class MoGeInferenceEngine:
    def __init__(self, model_path, project_root, version="v2", device="cuda", fp16=True):
        self._setup_env(project_root)
        self.device = self._get_optimal_device(device)
        self.fp16 = fp16
        
        # 只在第一个进程打印详细加载信息，避免刷屏
        if "shard_id" not in globals() or globals().get("shard_id", 0) == 0:
            print(f"📦 [系统] Loading Model on {self.device}...")

        try:
            from moge.model import import_model_class_by_version
            ModelClass = import_model_class_by_version(version)
            self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
            if self.fp16:
                self.model.half()
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e

    def _setup_env(self, project_root):
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            import moge
            import utils3d
        except ImportError:
            print(f"❌ 无法导入 MoGe/utils3d，请检查环境。")
            sys.exit(1)

    def _get_optimal_device(self, requested_device):
        if "cuda" in requested_device and "CUDA_VISIBLE_DEVICES" in os.environ:
            return torch.device("cuda:0")
        return torch.device(requested_device)

    def process_scene(self, input_dir, output_dir, config):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in VALID_IMG_EXTS])

        if not image_files: return

        # 这里的 tqdm 为了多进程显示干净，可以关掉或者简化
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
                logging.error(f"Failed {img_path}: {e}")

    def _infer_single_image(self, img_path, save_dir, config):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None: return

        h, w = image_bgr.shape[:2]
        process_bgr = image_bgr
        resize_to = config.get('resize')
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
            fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics) # 确保这里是 numpy
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

# ================= 🚀 主程序 =================

def main():
    args = setup_args()
    
    # 注入全局变量供 logger 使用
    global shard_id
    shard_id = args.shard_id

    # 1. 扫描任务
    CANDIDATE_IMG_DIRS = ["rgbs", "images_downsampled", "images", "img", "rgb", "image"]
    VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png')
    
    tasks = []
    if not os.path.exists(args.input_root): return

    for scene_name in sorted(os.listdir(args.input_root)):
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

    # 2. 初始化引擎
    config = {"resize": args.resize, "overwrite": args.overwrite, "save_maps": True}
    engine = MoGeInferenceEngine(args.model_path, args.project_root, device=args.device, fp16=True)

    # 3. 执行 (🔥🔥🔥 并行核心逻辑 🔥🔥🔥)
    my_tasks_count = 0
    for i, task in enumerate(tasks):
        # 核心：只处理属于当前 ID 的任务
        if i % args.total_shards != args.shard_id:
            continue
            
        print(f"[Worker {args.shard_id}] 处理场景: {os.path.basename(task['output'])}")
        engine.process_scene(task['input'], task['output'], config)
        my_tasks_count += 1
        
        if my_tasks_count % 5 == 0: torch.cuda.empty_cache()

    print(f"🎉 [Worker {args.shard_id}] 完成！共处理 {my_tasks_count} 个场景。")

if __name__ == "__main__":
    main()