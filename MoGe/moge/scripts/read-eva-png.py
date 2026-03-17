import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Depth Map Formats")
    # 修改这里为你的实际数据根目录
    parser.add_argument("--root_dir", type=str, default="/data1/szq/data/eval", help="数据集根目录")
    parser.add_argument("--samples_per_dataset", type=int, default=10, help="每个数据集抽查多少张")
    return parser.parse_args()

def get_dataset_name(file_path, root_dir):
    """
    假设目录结构是 root_dir/DatasetName/category/scene/...
    提取 DatasetName
    """
    rel_path = os.path.relpath(file_path, root_dir)
    return rel_path.split(os.sep)[0]

def analyze_depth_image(path):
    """
    读取并分析单张图片的格式和数值分布
    """
    # 必须使用 IMREAD_UNCHANGED 以保留 16-bit 深度信息
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return None
    
    stats = {
        "dtype": str(img.dtype),
        "shape": img.shape,
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        # 99分位数，过滤掉极值/无效值(如65535)的影响
        "p99": float(np.percentile(img, 99)) 
    }
    
    return stats

def main():
    args = parse_args()
    
    print(f"🔍 正在扫描目录: {args.root_dir} ...")
    
    # 1. 寻找所有 PNG 深度图
    dataset_files = defaultdict(list)
    
    # 遍历文件
    # 限制：为了速度，我们只找文件名包含 'depth' 且是 .png 的
    for root, dirs, files in os.walk(args.root_dir):
        for f in files:
            if f.endswith(".png") and "depth" in f.lower():
                full_path = os.path.join(root, f)
                d_name = get_dataset_name(full_path, args.root_dir)
                dataset_files[d_name].append(full_path)

    print(f"✅ 发现 {len(dataset_files)} 个数据集目录。开始抽样分析...\n")
    print(f"{'='*100}")
    print(f"{'Dataset':<15} | {'Type':<8} | {'Min':<6} | {'Max':<8} | {'P99 (稳定最大值)':<15} | {'猜测单位/缩放'}")
    print(f"{'='*100}")

    datasets_config_suggestion = {}

    for d_name, file_list in dataset_files.items():
        # 随机抽样
        samples = random.sample(file_list, min(len(file_list), args.samples_per_dataset))
        
        dtypes = []
        max_vals = []
        p99_vals = []
        
        for p in samples:
            stats = analyze_depth_image(p)
            if stats:
                dtypes.append(stats['dtype'])
                max_vals.append(stats['max'])
                p99_vals.append(stats['p99'])
        
        if not dtypes:
            print(f"{d_name:<15} | ❌ 无法读取或为空")
            continue

        # 统计聚合
        common_dtype = max(set(dtypes), key=dtypes.count)
        avg_max = np.mean(max_vals)
        avg_p99 = np.mean(p99_vals)
        
        # --- 智能推断逻辑 ---
        guess = "未知"
        scale = 1.0
        
        if "uint16" in common_dtype:
            if avg_p99 > 10000:
                guess = "毫米 (mm) -> /1000.0"
                scale = 1000.0
            elif avg_p99 > 1000:
                # 可能是毫米，但也可能是 KITTI 的 /256
                # KITTI 特征: 很多值在 5000-20000 之间 (20m - 80m)
                if "KITTI" in d_name.upper():
                    guess = "KITTI格式 -> /256.0"
                    scale = 256.0
                else:
                    guess = "毫米 (mm) -> /1000.0"
                    scale = 1000.0
            elif avg_p99 < 255:
                guess = "⚠️ 数值过小，可能是米(m)或视差?"
                scale = 1.0
            else:
                guess = "可能是厘米? 或特殊缩放"
                scale = 1000.0 # 默认猜测
        elif "uint8" in common_dtype:
            guess = "⚠️ 8位深度，精度极低"
            scale = 1.0 # 无法猜测
            
        # 针对具体数据集的硬编码修正建议
        if "DIODE" in d_name.upper():
            guess += " (DIODE原生通常是NPY, PNG可能是掩码或转换版)"
        if "ETH3D" in d_name.upper(): 
            guess += " (注意检查 readme)"
            
        print(f"{d_name:<15} | {common_dtype:<8} | {int(np.min(max_vals)):<6} | {int(avg_max):<8} | {avg_p99:<15.1f} | {guess}")
        
        datasets_config_suggestion[d_name] = scale

    print(f"{'='*100}")
    print("\n💡 建议后续 Step 3 代码中的 DATASET_SCALES 配置如下：")
    print("DATASET_SCALES = {")
    for k, v in datasets_config_suggestion.items():
        print(f"    '{k}': {v},")
    print("}")

if __name__ == "__main__":
    main()