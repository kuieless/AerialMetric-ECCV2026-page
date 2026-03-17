import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 源数据根目录
SOURCE_DIR = "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all-duodebufen"
# 输出数据根目录 (会自动创建)
TARGET_DIR = "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all"

# 采样间隔配置 (Scene Name: Stride)
# Stride = N 表示每 N 张取 1 张 (例如: [0, 1, 2, 3] -> stride=2 -> [0, 2])
SAMPLING_CONFIG = {
    "SYS": 4,           # 隔3张抽一张 (每4张取1张)
    "lower": 2,         # 隔1张抽一张 (每2张取1张)
    "park13": 2,        # 隔1张抽一张
    "sziit": 4,         # 隔3张抽一张
    "town1": 5,         # 隔4张抽一张
    "town2": 3,         # 隔2张抽一张
    "town3": 3,         # 隔2张抽一张
    "yuehai": 7,        # 隔6张抽一张
    "yingrenshi1": 1,   
    "yingrenshi2": 1,   
    # 其他未列出的场景默认 stride=1 (全部复制)
}

# 文件扩展名
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
NPY_EXT = '.npy'
# ===========================================

def get_file_pairs(scene_name, src_root):
    """
    获取场景下成对的 (Image, NPY) 文件列表，并按文件名排序
    """
    img_dir = src_root / f"{scene_name}-images"
    npy_dir = src_root / f"{scene_name}-npy"
    
    if not img_dir.exists() or not npy_dir.exists():
        return []

    # 获取所有图片
    img_files = sorted([
        f for f in os.listdir(img_dir) 
        if any(f.lower().endswith(ext) for ext in IMG_EXTS)
    ])
    
    pairs = []
    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        npy_name = base_name + NPY_EXT
        npy_path = npy_dir / npy_name
        
        if npy_path.exists():
            pairs.append({
                'base_name': base_name,
                'img_path': img_dir / img_name,
                'npy_path': npy_path
            })
            
    return pairs

def process_downsampling():
    src_root = Path(SOURCE_DIR)
    tgt_root = Path(TARGET_DIR)
    
    if not src_root.exists():
        print(f"❌ 源目录不存在: {src_root}")
        return

    # 1. 扫描所有场景
    all_subdirs = os.listdir(src_root)
    scenes = set()
    for d in all_subdirs:
        if d.endswith("-images"):
            scenes.add(d[:-7])
            
    print(f"📦 开始处理数据...")
    print(f"📂 源目录: {src_root}")
    print(f"📂 目标目录: {tgt_root}\n")
    
    total_copied = 0
    
    # 2. 遍历每个场景
    for scene in sorted(list(scenes)):
        # 获取采样步长
        stride = SAMPLING_CONFIG.get(scene, 1)
        
        # 获取成对文件
        pairs = get_file_pairs(scene, src_root)
        if not pairs:
            print(f"⚠️  {scene}: 未找到成对文件，跳过")
            continue
            
        # 执行采样
        # slice notation: list[start:stop:step]
        sampled_pairs = pairs[0::stride]
        
        print(f"🔄 处理 {scene:<15} | 总数: {len(pairs):<4} | 步长: {stride:<2} | 抽取后: {len(sampled_pairs):<4}")
        
        # 准备目标文件夹
        tgt_img_dir = tgt_root / f"{scene}-images"
        tgt_npy_dir = tgt_root / f"{scene}-npy"
        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        tgt_npy_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        for item in tqdm(sampled_pairs, desc=f"   Copying {scene}", leave=False):
            shutil.copy2(item['img_path'], tgt_img_dir / item['img_path'].name)
            shutil.copy2(item['npy_path'], tgt_npy_dir / item['npy_path'].name)
            total_copied += 1
            
    print("\n" + "="*50)
    print(f"✅ 处理完成！")
    print(f"📊 共处理 {len(scenes)} 个场景")
    print(f"📄 共复制 {total_copied} 对文件 (Image + NPY)")
    print(f"💾 输出位置: {tgt_root}")

if __name__ == "__main__":
    # 二次确认防止误操作
    print("⚠️  即将开始抽样复制，这可能会占用大量磁盘空间。")
    ans = input(f"确认将数据从\n{SOURCE_DIR}\n抽样复制到\n{TARGET_DIR}\n吗? (y/n): ")
    if ans.lower() == 'y':
        process_downsampling()
    else:
        print("已取消。")