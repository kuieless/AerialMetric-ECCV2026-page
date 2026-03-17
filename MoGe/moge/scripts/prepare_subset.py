import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True, help="原始大数据库根目录")
    parser.add_argument("--dst_root", type=str, required=True, help="存放抽取后数据的目录")
    parser.add_argument("--ratio", type=float, default=0.2, help="抽取比例")
    args = parser.parse_args()

    src_path = Path(args.src_root)
    dst_path = Path(args.dst_root)
    
    print(f"🔍 正在扫描深层目录: {src_path}")
    
    # 1. 深度扫描 (os.walk 会自动钻进每一层文件夹，不管是 CAMERA_06 还是 09)
    samples = []
    for root, dirs, files in os.walk(src_path):
        # 只要这一层同时有 image 和 depth，就是一个样本
        if "image.jpg" in files and "depth.png" in files:
            samples.append(Path(root))
            
    if not samples:
        print("❌ 没找到数据！请确认文件名是 image.jpg 和 depth.png")
        return

    # 2. 随机抽取
    random.shuffle(samples)
    count = int(len(samples) * args.ratio)
    selected = samples[:count]
    
    print(f"✅ 扫描到 {len(samples)} 个样本，抽取 {count} 个。")
    print(f"📂 正在扁平化目录结构...")

    # 3. 复制并重构 (关键步骤！)
    for src_dir in tqdm(selected):
        # 计算相对路径并替换斜杠，例如: DDAD/val/000065/CAMERA_06 -> DDAD_val_000065_CAMERA_06
        rel = src_dir.relative_to(src_path)
        flat_name = str(rel).replace(os.sep, "_")
        
        # 目标场景目录
        scene_dir = dst_path / flat_name
        
        # 🔥【关键修改】创建 images 子目录，这是 step1 推理代码强制要求的
        img_subdir = scene_dir / "images"
        img_subdir.mkdir(parents=True, exist_ok=True)
        
        # 复制 RGB 进 images 文件夹
        shutil.copy2(src_dir / "image.jpg", img_subdir / "image.jpg")
        
        # 复制 GT 放外面 (方便区分)
        shutil.copy2(src_dir / "depth.png", scene_dir / "depth_gt.png")

    print("🎉 数据准备完成！格式已适配。")

if __name__ == "__main__":
    main()