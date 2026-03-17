import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import Counter

def get_image_hw(img_path):
    """读取图像并返回 (H, W)"""
    try:
        img = cv2.imread(img_path)
        if img is None: return None
        return img.shape[:2]
    except:
        return None

def get_npy_shape(npy_path):
    """读取 NPY 头部并返回 (H, W)"""
    try:
        # mmap_mode='r' 只读头部，速度快
        arr = np.load(npy_path, mmap_mode='r')
        # 处理 (H, W) 或 (1, H, W) 或 (H, W, 1)
        if arr.ndim == 2:
            return arr.shape
        elif arr.ndim == 3:
            # 假设 shape 是 [C, H, W] 或 [H, W, C]
            # 通常 H, W 是较大的数值，C 是 1 或 3
            # 这里简单取最大的两个维度作为 H, W (并不严谨但够用)
            dims = sorted(arr.shape, reverse=True)
            return (dims[0], dims[1]) # 返回大的两个维度
        return arr.shape
    except:
        return None

def find_outliers(target_dir, file_type='image'):
    root = Path(target_dir)
    if not root.exists():
        print(f"❌ 路径不存在: {root}")
        return

    print(f"正在扫描: {root} ...")
    
    # 1. 收集所有文件的形状
    files_stats = []
    exts = ['.jpg', '.png', '.jpeg'] if file_type == 'image' else ['.npy']
    
    all_files = sorted([
        f for f in os.listdir(root) 
        if any(f.lower().endswith(ext) for ext in exts)
    ])
    
    if not all_files:
        print("❌ 文件夹为空或无匹配文件。")
        return

    shape_counter = Counter()

    for f in all_files:
        f_path = str(root / f)
        shape = None
        
        if file_type == 'image':
            shape = get_image_hw(f_path)
        else:
            shape = get_npy_shape(f_path)
            
        if shape:
            files_stats.append({'name': f, 'shape': shape})
            shape_counter[shape] += 1
        else:
            print(f"⚠️ 无法读取: {f}")

    # 2. 找出“主流”形状 (数量最多的那个)
    if not shape_counter:
        return

    majority_shape = shape_counter.most_common(1)[0][0]
    total_count = len(files_stats)
    majority_count = shape_counter[majority_shape]
    
    print(f"\n📊 统计结果 ({file_type}):")
    print(f"   - 总文件数: {total_count}")
    print(f"   - 主流形状: {majority_shape} (占比 {majority_count}/{total_count})")
    
    if total_count == majority_count:
        print("✅ 所有文件形状一致，没有异常值。")
        return

    # 3. 列出“异类”
    print(f"\n🚨 发现 {total_count - majority_count} 个分辨率异常的文件：")
    print("-" * 60)
    print(f"{'文件名':<40} | {'形状'}")
    print("-" * 60)
    
    outlier_paths = []
    
    for item in files_stats:
        if item['shape'] != majority_shape:
            print(f"{item['name']:<40} | {item['shape']}")
            outlier_paths.append(str(root / item['name']))
            
    print("-" * 60)
    
    # 4. (可选) 生成删除命令
    print("\n💡 提示: 如果你想删除这些文件，可以使用以下命令 (请谨慎操作):")
    print("# 复制下方命令到终端执行")
    print(f"rm {' '.join(outlier_paths)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="要检查的具体文件夹路径 (例如 .../park13-images)")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "img", "npy"], help="文件类型: img 或 npy")
    args = parser.parse_args()
    
    # 自动判断类型
    check_type = args.type
    if check_type == "auto":
        if "npy" in args.path.lower():
            check_type = "npy"
        else:
            check_type = "image"
            
    find_outliers(args.path, check_type)

    #python /home/data1/szq/Megadepth/benchemarkdata/Acodes/findpark13.py /home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/park13-images