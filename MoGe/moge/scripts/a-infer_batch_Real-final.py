

import argparse
import os
import sys
import json
import time
import logging
import traceback
import numpy as np

# 
# ========================================================
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib

# 设置无头模式，防止服务器报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 🔧 配置与日志 =================

# 设置简单的日志记录
logging.basicConfig(
    filename='inference_error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_args():
    parser = argparse.ArgumentParser(description="MoGe Inference Script (Robust)")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--input_root", type=str, required=True, help="数据集输入根目录")
    parser.add_argument("--output_root", type=str, required=True, help="结果输出根目录")
    parser.add_argument("--project_root", type=str, default="/home/szq/moge2/MoGe", help="MoGe项目根目录")
    
    # 新增参数
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备 (例如 cuda:0, cuda:1)")
    parser.add_argument("--resize", type=int, default=None, help="缩放长边大小 (例如 1024)")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的结果 (默认跳过)")
    
    return parser.parse_args()

# ================= 🧠 推理引擎 =================

class MoGeInferenceEngine:
    def __init__(self, model_path, project_root, version="v2", device="cuda", fp16=True):
        self._setup_env(project_root)
        
        # --- 🛡️ 核心修复：智能设备选择 ---
        self.device = self._get_optimal_device(device)
        self.fp16 = fp16
        
        print(f"\n📦 [系统] 正在加载模型...")
        print(f"   Path: {model_path}")
        print(f"   Device: {self.device} (User requested: {device})")

        try:
            from moge.model import import_model_class_by_version
            ModelClass = import_model_class_by_version(version)
            self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
            if self.fp16:
                self.model.half()
            print("✅ [系统] 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e

    def _setup_env(self, project_root):
        """确保能导入 MoGe 包"""
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            import moge
            import utils3d
        except ImportError:
            print(f"❌ 无法导入 MoGe，请检查路径: {project_root}")
            sys.exit(1)

    def _get_optimal_device(self, requested_device):
        """
        🛡️ 解决 Error 101 的核心逻辑
        如果用户设置了 CUDA_VISIBLE_DEVICES，强制使用 cuda:0
        """
        if "cuda" in requested_device and "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            print(f"⚠️ 检测到环境变量 CUDA_VISIBLE_DEVICES={visible_devices}")
            print(f"   强制将设备重映射为 'cuda:0' 以匹配 PyTorch 的视图。")
            return torch.device("cuda:0")
        return torch.device(requested_device)

    def process_scene(self, input_dir, output_dir, config):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in VALID_IMG_EXTS])

        if not image_files:
            return

        print(f"▶️  处理: {input_path.name} ({len(image_files)} imgs)")

        for img_path in tqdm(image_files, desc="   Processing", leave=False):
            # 准备保存路径
            rel_path = img_path.relative_to(input_path)
            # 扁平化还是保持结构？这里保持原始结构，但把文件名作为文件夹
            # 你的逻辑: output / (结构) / (文件名_stem) / results
            save_dir = output_path / rel_path.parent / img_path.stem
            
            # --- 🛡️ 断点续传：检查是否已存在 ---
            if not config.get('overwrite', False):
                if (save_dir / 'depth.npy').exists():
                    # print(f"Skipping {img_path.name}") 
                    continue

            try:
                self._infer_single_image(img_path, save_dir, config)
            except torch.cuda.OutOfMemoryError:
                print(f"💥 [显存不足] 跳过大图: {img_path.name}")
                logging.error(f"OOM Error: {img_path}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ [错误] 处理图片失败: {img_path.name}")
                logging.error(f"Failed {img_path}: {e}")

    def _infer_single_image(self, img_path, save_dir, config):
        # 读取图片
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logging.warning(f"无法读取图片: {img_path}")
            return

        h, w = image_bgr.shape[:2]
        
        # 缩放逻辑
        process_bgr = image_bgr
        resize_to = config.get('resize')
        if resize_to is not None:
            scale = resize_to / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                # 确保是14的倍数 (ViT patch size通常为14)
                new_w = (new_w // 14) * 14
                new_h = (new_h // 14) * 14
                process_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 转 Tensor
        image_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).to(self.device)
        
        if self.fp16:
            image_tensor = image_tensor.half()

        # 推理
        with torch.inference_mode():
            output = self.model.infer(
                image_tensor,
                resolution_level=config.get('resolution_level', 9),
                use_fp16=self.fp16
            )

        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果
        cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr) # 存原图方便对比
        
        depth = output['depth'].cpu().numpy()
        np.save(str(save_dir / 'depth.npy'), depth)

        # 保存 Scale
        if 'metric_scale' in output:
            with open(save_dir / 'scale.json', 'w') as f:
                json.dump({'metric_scale': output['metric_scale'].float().cpu().item()}, f, indent=4)

        # 保存 FOV
        if 'intrinsics' in output:
            import utils3d # 延迟导入
            intrinsics = output['intrinsics'].cpu().numpy()
            fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
            with open(save_dir / 'fov.json', 'w') as f:
                json.dump({
                    'fov_x': round(float(np.rad2deg(fov_x)), 2),
                    'fov_y': round(float(np.rad2deg(fov_y)), 2)
                }, f, indent=4)

        # 可视化
        if config.get('save_maps', True):
            self._save_visualization(depth, save_dir / 'depth_vis.png')

    def _save_visualization(self, depth, save_path):
        valid_mask = (depth > 0) & np.isfinite(depth)
        if valid_mask.sum() == 0: return

        vmin = np.percentile(depth[valid_mask], 2)
        vmax = np.percentile(depth[valid_mask], 98)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(depth, cmap='Spectral', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=80, bbox_inches='tight')
        plt.close()

# ================= 🚀 主程序 =================

def main():
    args = setup_args()

    # 1. 扫描任务
    CANDIDATE_IMG_DIRS = ["rgbs", "images_downsampled", "images", "img", "rgb", "image"]
    VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png') # 简化检查
    
    tasks = []
    if not os.path.exists(args.input_root):
        print(f"❌ 数据集不存在: {args.input_root}")
        return

    print(f"📂 扫描: {args.input_root}")
    for scene_name in sorted(os.listdir(args.input_root)):
        scene_path = os.path.join(args.input_root, scene_name)
        if not os.path.isdir(scene_path): continue
        
        # 查找图片子目录
        input_rgbs_path = None
        for candidate in CANDIDATE_IMG_DIRS:
            p = os.path.join(scene_path, candidate)
            if os.path.isdir(p) and len(os.listdir(p)) > 0:
                input_rgbs_path = p
                break
        
        # 如果没找到标准名，但根目录下就有图片
        if not input_rgbs_path:
             # 简单检查是否有图
             if any(f.endswith(VALID_IMG_EXTS) for f in os.listdir(scene_path)):
                 input_rgbs_path = scene_path

        if input_rgbs_path:
            tasks.append({
                "input": input_rgbs_path,
                "output": os.path.join(args.output_root, scene_name)
            })

    print(f"✅ 发现 {len(tasks)} 个场景")

    # 2. 初始化
    config = {
        "resize": args.resize,
        "overwrite": args.overwrite,
        "save_maps": True,
        "resolution_level": 9
    }

    engine = MoGeInferenceEngine(
        model_path=args.model_path,
        project_root=args.project_root,
        device=args.device, # 这里传入用户想要的 cuda:1
        fp16=True
    )

    # 3. 执行
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Scn: {os.path.basename(task['output'])}")
        engine.process_scene(task['input'], task['output'], config)
        
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache() # 定期清理

    print("\n🎉 完成！")

if __name__ == "__main__":
    main()