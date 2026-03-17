import os
import random
import csv
import argparse
import shutil
from tqdm import tqdm
from pathlib import Path

# ================= 配置区 =================
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# 🔍 核心修改：文件名黑名单
# 如果图片文件名包含这些字符串，绝对不是我们要推理的 RGB 图
FILENAME_BLACKLIST = [
    'depth', 'groundtruth', 'segmentation', 'mask', 
    'normal', 'semseg', 'confidence', 'validity'
]

# ========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Step 0: Preprocess and Sample")
    parser.add_argument("--source_root", type=str, default="/data1/szq/data/eval", help="原始数据根目录")
    parser.add_argument("--output_root", type=str, default="/data1/szq/data/benchmark_staging", help="输出目录")
    
    # 采样控制
    parser.add_argument("--sample_rate", type=float, default=0.20, help="采样率 (0.0 ~ 1.0)")
    parser.add_argument("--all", action="store_true", help="是否使用全部数据 (开启则忽略 sample_rate)")
    
    return parser.parse_args()

def is_depth_file(filename):
    """判断是否为深度真值文件"""
    fname = filename.lower()
    # 必须是 png 且包含 depth 或 groundtruth
    # 注意：根据你的数据集情况，这里可能需要微调
    return filename.endswith('.png') and ('depth' in fname or 'groundtruth' in fname)

def find_sibling_image(depth_path):
    """
    根据深度图路径寻找对应的 RGB 图
    增强了过滤逻辑，防止选到 segmentation.png
    """
    parent = os.path.dirname(depth_path)
    try:
        files = os.listdir(parent)
    except Exception:
        return None

    depth_name = os.path.basename(depth_path)
    
    # 1. 初步筛选：是图片，且不是深度图本身
    candidates = [f for f in files if f.lower().endswith(IMG_EXTS) and f != depth_name]
    
    # 2. 🔥 核心修改：黑名单过滤
    # 排除所有包含 segmentation, mask 等关键词的文件
    valid_imgs = []
    for c in candidates:
        c_lower = c.lower()
        if not any(bad_word in c_lower for bad_word in FILENAME_BLACKLIST):
            valid_imgs.append(c)
    
    if not valid_imgs:
        return None

    # 3. 匹配逻辑
    # 情况 A: 只有一个合法的 RGB 文件 (最理想)
    if len(valid_imgs) == 1:
        return os.path.join(parent, valid_imgs[0])
    
    # 情况 B: 有多个文件，尝试通过文件名匹配
    # 例如 depth: scene_01_depth.png -> 寻找 scene_01.jpg
    depth_stem = os.path.splitext(depth_name)[0]
    # 去掉常见的后缀
    clean_stem = depth_stem.replace('_depth', '').replace('depth', '').replace('_groundtruth', '')
    
    # 精确匹配尝试
    for img in valid_imgs:
        img_stem = os.path.splitext(img)[0]
        if img_stem == clean_stem:
            return os.path.join(parent, img)
            
    # 模糊匹配尝试 (RGB文件名包含在深度文件名里，或者反之)
    for img in valid_imgs:
        if clean_stem in img or os.path.splitext(img)[0] in depth_name:
            return os.path.join(parent, img)
            
    # 如果实在匹配不到，且有多个候选，打印警告并跳过 (宁缺毋滥)
    # print(f"⚠️ Warning: Ambiguous images for {depth_name}: {valid_imgs}")
    return None

def get_dataset_name(path, source_root):
    rel = os.path.relpath(path, source_root)
    return rel.split(os.sep)[0]

def main():
    args = parse_args()
    
    # 定义 staging 目录
    staging_img_dir = os.path.join(args.output_root, "images", "mixed_val")
    csv_path = os.path.join(args.output_root, "benchmark_index.csv")

    # 1. 🧹 自动清理旧数据
    # 只要运行这个脚本，就会把旧的软链接删掉，确保不会残留 segmentation.png 这种错误文件
    if os.path.exists(staging_img_dir):
        print(f"🧹 正在清理旧目录: {staging_img_dir}")
        shutil.rmtree(staging_img_dir)
    os.makedirs(staging_img_dir, exist_ok=True)
    
    print(f"🔍 正在扫描: {args.source_root} ...")
    dataset_files = {} 
    
    # 2. 扫描文件
    for root, _, files in os.walk(args.source_root):
        depth_files = [f for f in files if is_depth_file(f)]
        for d_file in depth_files:
            d_path = os.path.join(root, d_file)
            i_path = find_sibling_image(d_path)
            
            if i_path:
                d_name = get_dataset_name(d_path, args.source_root)
                if d_name not in dataset_files: dataset_files[d_name] = []
                dataset_files[d_name].append((d_path, i_path))

    # 3. 采样或全量
    if args.all:
        print(f"✅ 扫描完成。模式: [全量数据] (忽略 sample_rate)")
    else:
        print(f"✅ 扫描完成。模式: [随机采样] Rate: {args.sample_rate}")
        
    total_count = 0
    csv_rows = []
    
    for d_name, pairs in dataset_files.items():
        if args.all:
            k = len(pairs)
        else:
            k = max(1, int(len(pairs) * args.sample_rate))
            
        sampled = random.sample(pairs, k)
        print(f"   - {d_name:<10}: 总数 {len(pairs):<6} -> 选取 {k}")
        
        for gt_path, img_path in tqdm(sampled, desc=f"Linking {d_name}", leave=False):
            # 生成唯一且安全的文件名
            rel_path = os.path.relpath(img_path, args.source_root)
            # 把路径分隔符变成下划线，打平文件名
            safe_name = rel_path.replace(os.sep, '_').replace(' ', '')
            
            # 确保后缀名正确 (有些时候源文件没有后缀或后缀大写)
            ext = os.path.splitext(img_path)[1].lower()
            if not ext: ext = '.jpg'
            if not safe_name.lower().endswith(IMG_EXTS):
                safe_name += ext
                
            dst_link = os.path.join(staging_img_dir, safe_name)
            
            # 创建软链接 (Symlink)
            if os.path.exists(dst_link): os.remove(dst_link)
            os.symlink(img_path, dst_link)
            
            # 记录到 CSV
            csv_rows.append([
                'val', d_name, safe_name, img_path, gt_path
            ])
            total_count += 1
            
    # 4. 写入索引
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Split', 'Dataset', 'Renamed_Image', 'Original_Image', 'GT_Path'])
        writer.writerows(csv_rows)
        
    print(f"\n🎉 预处理完成! 共准备了 {total_count} 张图片。")
    print(f"📂 推理输入根目录: {os.path.dirname(staging_img_dir)}") 
    print(f"📄 索引文件:       {csv_path}")

if __name__ == "__main__":
    main()
    # import os
# import random
# import csv
# import argparse
# from tqdm import tqdm
# from pathlib import Path

# # ================= 配置区 =================
# IMG_EXTS = ('.jpg', '.jpeg', '.png')
# DEPTH_KEYWORDS = ['depth.png', 'groundtruth.png'] 
# # ========================================

# def parse_args():
#     parser = argparse.ArgumentParser(description="Step 0: Preprocess and Sample")
#     parser.add_argument("--source_root", type=str, default="/data1/szq/data/eval", help="原始数据根目录")
#     parser.add_argument("--output_root", type=str, default="/data1/szq/data/benchmark_staging", help="输出目录")
#     parser.add_argument("--sample_rate", type=float, default=0.15, help="采样率")
#     return parser.parse_args()

# def is_depth_file(filename):
#     fname = filename.lower()
#     return filename.endswith('.png') and ('depth' in fname or 'groundtruth' in fname)

# def find_sibling_image(depth_path):
#     parent = os.path.dirname(depth_path)
#     files = os.listdir(parent)
#     candidates = [f for f in files if f.lower().endswith(IMG_EXTS) and f != os.path.basename(depth_path)]
#     valid_imgs = [c for c in candidates if 'depth' not in c.lower()]
    
#     if len(valid_imgs) == 1:
#         return os.path.join(parent, valid_imgs[0])
    
#     depth_stem = os.path.splitext(os.path.basename(depth_path))[0].replace('_depth', '').replace('depth', '')
#     for img in valid_imgs:
#         if depth_stem in img:
#             return os.path.join(parent, img)
#     return None

# def get_dataset_name(path, source_root):
#     rel = os.path.relpath(path, source_root)
#     return rel.split(os.sep)[0]

# def main():
#     args = parse_args()
    
#     # 🔥 核心修改：增加一层子目录 'mixed_val'，防止 Step 1 读到根目录文件报错
#     staging_img_dir = os.path.join(args.output_root, "images", "mixed_val")
    
#     # 清空旧数据以防混淆
#     if os.path.exists(staging_img_dir):
#         import shutil
#         print(f"🧹 清理旧目录: {staging_img_dir}")
#         shutil.rmtree(staging_img_dir)
#     os.makedirs(staging_img_dir, exist_ok=True)
    
#     csv_path = os.path.join(args.output_root, "benchmark_index.csv")
    
#     print(f"🔍 正在扫描: {args.source_root} ...")
#     dataset_files = {} 
    
#     for root, _, files in os.walk(args.source_root):
#         depth_files = [f for f in files if is_depth_file(f)]
#         for d_file in depth_files:
#             d_path = os.path.join(root, d_file)
#             i_path = find_sibling_image(d_path)
#             if i_path:
#                 d_name = get_dataset_name(d_path, args.source_root)
#                 if d_name not in dataset_files: dataset_files[d_name] = []
#                 dataset_files[d_name].append((d_path, i_path))

#     print(f"✅ 扫描完成。开始采样 (Rate: {args.sample_rate})...")
#     total_count = 0
#     csv_rows = []
    
#     for d_name, pairs in dataset_files.items():
#         k = max(1, int(len(pairs) * args.sample_rate))
#         sampled = random.sample(pairs, k)
#         print(f"   - {d_name:<10}: 采样 {k}")
        
#         for gt_path, img_path in tqdm(sampled, desc=f"Linking {d_name}", leave=False):
#             rel_path = os.path.relpath(img_path, args.source_root)
#             safe_name = rel_path.replace(os.sep, '_').replace(' ', '')
            
#             if not safe_name.lower().endswith('.jpg'):
#                 safe_name = os.path.splitext(safe_name)[0] + '.jpg'
                
#             dst_link = os.path.join(staging_img_dir, safe_name)
            
#             if os.path.exists(dst_link): os.remove(dst_link)
#             os.symlink(img_path, dst_link)
            
#             csv_rows.append([
#                 'val', d_name, safe_name, img_path, gt_path
#             ])
#             total_count += 1
            
#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Split', 'Dataset', 'Renamed_Image', 'Original_Image', 'GT_Path'])
#         writer.writerows(csv_rows)
        
#     print(f"\n🎉 预处理完成!")
#     # 注意：这里我们返回上一级目录，作为 Step 1 的输入
#     # 这样 Step 1 看到的结构是: input_root/mixed_val/image.jpg
#     print(f"📂 推理输入根目录: {os.path.dirname(staging_img_dir)}") 
#     print(f"📄 索引文件:       {csv_path}")

# if __name__ == "__main__":
#     main()