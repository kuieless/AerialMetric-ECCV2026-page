# import os
# import json
# import numpy as np
# import cv2
# from pathlib import Path
# from tqdm import tqdm
# from moge.utils.io import read_depth, read_image

# # ================= 配置 =================
# # 指向你的内存盘路径
# DATASET_ROOT = "/dev/shm/szq_moge2"
# # 索引文件路径
# INDEX_FILE = "/dev/shm/szq_moge2/.index.txt" 
# # =======================================

# def test_load():
#     print(f"正在读取索引: {INDEX_FILE}")
#     with open(INDEX_FILE, 'r') as f:
#         lines = [line.strip() for line in f.readlines() if line.strip()]

#     print(f"共发现 {len(lines)} 个样本，开始扫雷...")
    
#     # 使用单线程顺序检查，这样一旦崩溃，最后打印出的那个就是凶手
#     for idx, line in enumerate(tqdm(lines)):
#         # 解析路径
#         # 假设 index 格式是: scene_name/image_name
#         # 需要根据你实际的 index 格式调整 split
#         parts = line.split()
#         rel_path = parts[0]
        
#         full_path = os.path.join(DATASET_ROOT, rel_path)
        
#         image_path = os.path.join(full_path, "image.jpg")
#         depth_path = os.path.join(full_path, "depth.png") # 检查 PNG
        
#         # 1. 打印当前检查的文件 (为了在崩溃时知道是谁)
#         # 加上 flush=True 确保立即输出到终端
#         # print(f"Checking: {rel_path}", end='\r', flush=True)

#         try:
#             # === 测试 RGB 读取 ===
#             if not os.path.exists(image_path):
#                 print(f"\n❌ 图片缺失: {image_path}")
#                 continue
#             img = read_image(image_path)
#             if img is None:
#                 print(f"\n❌ 图片损坏 (None): {image_path}")

#             # === 测试 深度图 读取 (重点！) ===
#             if not os.path.exists(depth_path):
#                 print(f"\n❌ 深度图缺失: {depth_path}")
#                 continue
            
#             # 这里调用官方的 read_depth，看看会不会崩
#             depth = read_depth(depth_path)
            
#             # 简单检查数值
#             if depth is None:
#                 print(f"\n❌ 深度图损坏 (None): {depth_path}")
#             if np.isnan(depth).all():
#                 print(f"\n⚠️ 深度图全为 NaN: {depth_path}")

#         except Exception as e:
#             print(f"\n❌ Python 报错: {rel_path} -> {e}")
        
#         # 如果是 SegFault，程序会直接退出，
#         # 你只需看终端里进度条停在哪里，或者上一行 print 是什么。

# if __name__ == "__main__":
#     # 禁用多线程，防止干扰调试
#     cv2.setNumThreads(0)
#     os.environ["OMP_NUM_THREADS"] = "1"
    
#     test_load()
#     print("\n✅ 所有数据检查通过！如果这里打印出来了，说明数据没问题。")

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ================= 配置 =================
# 你的内存盘数据集路径
DATASET_ROOT = "/dev/shm/szq_moge2"
# 索引文件路径 (确保它也在内存盘里)
INDEX_FILE = os.path.join(DATASET_ROOT, ".index.txt")
# =======================================

def check_image_cv2(path):
    # 检查 RGB 图片
    try:
        if not os.path.exists(path): return False, "File missing"
        if os.path.getsize(path) == 0: return False, "Zero size"
        
        # 尝试解码
        img = cv2.imread(path)
        if img is None: return False, "cv2 decode failed"
        return True, None
    except Exception as e:
        return False, str(e)

def check_depth_png(path):
    # 检查 16-bit 深度图 (你的新格式)
    try:
        if not os.path.exists(path): return False, "File missing"
        if os.path.getsize(path) == 0: return False, "Zero size"
        
        # 尝试用 PIL 读取 (必须能读元数据)
        try:
            pil_img = Image.open(path)
            pil_img.load() # 强制解码像素
            
            # 检查关键元数据 (near/far)
            if 'near' not in pil_img.info or 'far' not in pil_img.info:
                return False, "Missing Metadata (near/far)"
                
        except Exception as e:
            return False, f"PIL Error: {e}"

        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print(f"🔍 正在读取索引: {INDEX_FILE}")
    if not os.path.exists(INDEX_FILE):
        print("❌ 索引文件不存在！请检查路径。")
        return

    with open(INDEX_FILE, 'r') as f:
        lines = [line.strip().split()[0] for line in f.readlines() if line.strip()]

    print(f"📦 共发现 {len(lines)} 个样本，开始逐个排雷...")
    print("⚠️  如果脚本突然退出，最后打印的那个文件就是凶手！")

    bad_files = []

    # 使用单线程顺序检查
    for i, rel_path in enumerate(tqdm(lines)):
        # 构造路径
        sample_path = os.path.join(DATASET_ROOT, rel_path)
        img_path = os.path.join(sample_path, "image.jpg")
        depth_path = os.path.join(sample_path, "depth.png")

        # 1. 检查 RGB
        ok, msg = check_image_cv2(img_path)
        if not ok:
            print(f"\n❌ [BAD RGB] {img_path} -> {msg}")
            bad_files.append(rel_path)
            continue

        # 2. 检查 Depth
        ok, msg = check_depth_png(depth_path)
        if not ok:
            print(f"\n❌ [BAD DEPTH] {depth_path} -> {msg}")
            bad_files.append(rel_path)
            continue
            
    print("\n" + "="*30)
    if bad_files:
        print(f"💀 发现 {len(bad_files)} 个损坏样本。")
        print("建议从 .index.txt 中删除这些行，或者重新生成这些文件。")
        
        # 自动生成一个清洗后的 index
        clean_index_path = INDEX_FILE + ".clean"
        print(f"正在生成清洗后的索引: {clean_index_path}")
        with open(INDEX_FILE, 'r') as f_in, open(clean_index_path, 'w') as f_out:
            bad_set = set(bad_files)
            for line in f_in:
                key = line.strip().split()[0]
                if key not in bad_set:
                    f_out.write(line)
        print("✅ 已生成新索引。请修改 train.json 指向 .index.txt.clean")
    else:
        print("🎉 恭喜！所有图片都能正常读取，数据没有问题。")
        print("如果依然报错 -11，那确实是环境/库冲突问题。")

if __name__ == "__main__":
    main()