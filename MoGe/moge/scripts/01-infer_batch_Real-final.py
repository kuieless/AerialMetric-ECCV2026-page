

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import json
import itertools
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # 这是一个好习惯，服务器端绘图防报错
import matplotlib.pyplot as plt

# ================= ⚙️ 1. 用户配置区域 (根据你的环境修改) =================

# MoGe 项目的根目录 (用于 import moge 包)
PROJECT_ROOT = "/home/szq/moge2/MoGe"

# 模型权重路径
#全参数的
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/workspace/final-fintune2-1.18/checkpoint/00001000.pt"

#head的
# MODEL_PATH = "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-2/checkpoint/00016500_ema.pt"
MODEL_PATH = "/home/szq/moge2/MoGe/workspace/final-fintune2-127-lora1-r8/checkpoint/00000200.pt"

# 数据集输入根目录
DATASET_ROOT = "/data1/szq/data/Val2"

# 结果输出根目录
# OUTPUT_ROOT = "/home/data1/szq/Megadepth/becnmarkdata2/Val-Results-moge2-1000step"

#head的
OUTPUT_ROOT = "/data1/szq/data/Val-point-lora"
# 默认参数
DEFAULT_CONFIG = {
    "version": "v2",
    "device": "cuda",
    "fp16": True,
    "resize": None,       # 如果需要缩放，填整数，例如 1024
    "resolution_level": 9,
    "save_maps": True,    # 对应原参数 --maps
    "save_glb": False,
    "save_ply": True
}

# 候选图片文件夹名称
CANDIDATE_IMG_DIRS = ["rgbs", "images_downsampled", "images", "img", "rgb", "image"]
VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# ================= 🔧 2. 环境设置与导包 =================

# 动态添加 MoGe 路径到环境变量，确保能 import moge
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from moge.model import import_model_class_by_version
    import utils3d
except ImportError as e:
    print(f"❌ 无法导入 MoGe 库，请检查 PROJECT_ROOT 是否正确: {PROJECT_ROOT}")
    raise e

# ================= 🧠 3. 推理引擎类 (模型只加载一次) =================

class MoGeInferenceEngine:
    def __init__(self, model_path, version="v2", device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.version = version
        
        print(f"\n📦 [系统] 正在加载模型权重，请稍候...")
        print(f"   path: {model_path}")
        
        # 加载模型类
        ModelClass = import_model_class_by_version(version)
        # 加载权重
        self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
        
        if self.fp16:
            self.model.half()
            print("⚡ [系统] FP16 模式已开启")
        print("✅ [系统] 模型加载完成！准备就绪。\n")

    def process_scene(self, input_dir, output_dir, config):
        """处理单个场景文件夹"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 扫描图片
        image_files = sorted([
            p for p in input_path.rglob('*') 
            if p.suffix.lower() in VALID_IMG_EXTS
        ])
        
        if not image_files:
            print(f"⚠️ [跳过] 文件夹为空: {input_dir}")
            return

        print(f"▶️  正在处理: {input_path.name} ({len(image_files)} 张图片)")

        # 开始推理循环
        with torch.inference_mode():
            for img_path in tqdm(image_files, desc=f"   Processing", leave=False):
                self._infer_single_image(img_path, input_path, output_path, config)

    def _infer_single_image(self, img_path, root_input, root_output, config):
        # 读取图片
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return

        h, w = image_bgr.shape[:2]
        
        # 缩放逻辑
        process_bgr = image_bgr
        resize_to = config.get('resize')
        if resize_to is not None:
            scale = resize_to / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                new_w = (new_w // 14) * 14
                new_h = (new_h // 14) * 14
                process_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 转 Tensor
        image_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).to(self.device)
        
        if self.fp16:
            image_tensor = image_tensor.half()

        # 模型推理
        output = self.model.infer(
            image_tensor,
            resolution_level=config.get('resolution_level', 9),
            use_fp16=self.fp16
        )

        # 准备保存路径 (保持相对目录结构)
        rel_path = img_path.relative_to(root_input)
        save_dir = root_output / rel_path.parent / img_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- 数据提取与保存 ---
        
        # 1. 保存原图
        cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr)

        # 2. 保存深度 (numpy)
        depth = output['depth'].cpu().numpy()
        points = output['points'].cpu().numpy()
        np.save(str(save_dir / 'depth.npy'), depth)

        cv2.imwrite(
                str(save_dir / 'points.exr'), 
                cv2.cvtColor(points, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
            ) # <--- ADDED: Save points.exr

        # 3. 保存 Scale
        metric_scale = 1.0
        if 'metric_scale' in output:
            metric_scale = output['metric_scale'].float().cpu().item()
        
        with open(save_dir / 'scale.json', 'w') as f:
            json.dump({'metric_scale': metric_scale}, f, indent=4)

        # 4. 保存 FOV
        if 'intrinsics' in output:
            intrinsics = output['intrinsics'].cpu().numpy()
            fov_x_rad, fov_y_rad = utils3d.np.intrinsics_to_fov(intrinsics)
            with open(save_dir / 'fov.json', 'w') as f:
                json.dump({
                    'fov_x': round(float(np.rad2deg(fov_x_rad)), 2),
                    'fov_y': round(float(np.rad2deg(fov_y_rad)), 2)
                }, f, indent=4)

        # 5. 保存可视化 (如果开启)
        if config.get('save_maps'):
            self._save_visualization(depth, save_dir / 'depth_vis.png')

    def _save_visualization(self, depth, save_path):
        """简化的可视化保存，避免阻塞太久"""
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


# ================= 📂 4. 目录扫描辅助函数 =================

def has_files_with_ext(folder_path, extensions):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(extensions):
            return True
    return False

def find_valid_input_folder(scene_path):
    # 策略 A: 查表
    for candidate in CANDIDATE_IMG_DIRS:
        target_path = os.path.join(scene_path, candidate)
        if os.path.isdir(target_path) and has_files_with_ext(target_path, VALID_IMG_EXTS):
            return target_path
    # 策略 B: 盲搜一级子目录
    for item in os.listdir(scene_path):
        sub_path = os.path.join(scene_path, item)
        if os.path.isdir(sub_path) and has_files_with_ext(sub_path, VALID_IMG_EXTS):
            return sub_path
    return None

# ================= 🚀 5. 主执行逻辑 =================

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 数据集路径不存在: {DATASET_ROOT}")
        return

    # 1. 扫描所有任务
    tasks = []
    all_scenes = sorted(os.listdir(DATASET_ROOT))
    print(f"📂 正在扫描目录: {DATASET_ROOT}")

    for scene_name in all_scenes:
        scene_path = os.path.join(DATASET_ROOT, scene_name)
        if not os.path.isdir(scene_path): continue
        
        input_rgbs_path = find_valid_input_folder(scene_path)
        if input_rgbs_path:
            output_path = os.path.join(OUTPUT_ROOT, scene_name)
            tasks.append({
                "input": input_rgbs_path,
                "output": output_path
            })
        else:
            print(f"⏭️  [跳过] 未找到图片目录: {scene_name}")

    if not tasks:
        print("⚠️ 没有发现有效任务，退出。")
        return

    print(f"✅ 发现 {len(tasks)} 个场景待处理。\n")

    # 2. 初始化引擎 (只运行一次！)
    engine = MoGeInferenceEngine(
        model_path=MODEL_PATH,
        version=DEFAULT_CONFIG['version'],
        fp16=DEFAULT_CONFIG['fp16']
    )

    # 3. 批量处理
    for i, task in enumerate(tasks):
        print(f"--------------------------------------------------")
        print(f"[{i+1}/{len(tasks)}] 场景: {os.path.basename(task['output'])}")
        try:
            engine.process_scene(task['input'], task['output'], DEFAULT_CONFIG)
        except Exception as e:
            print(f"❌ 处理场景失败 {task['input']}: {e}")
            import traceback
            traceback.print_exc()
        
        # 可选：每处理完一个场景手动清一下缓存，防止显存碎片（对于大场景有用）
        torch.cuda.empty_cache()

    print("\n🎉🎉🎉 所有任务处理完毕！")

if __name__ == "__main__":
    main()