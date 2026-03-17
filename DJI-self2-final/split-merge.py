# import os
# import shutil
# import pandas as pd
# from tqdm import tqdm

# # ================= ⚙️ 配置区 =================

# # 1. 你的合并后的 CSV 路径 (请修改为你实际的 CSV 路径)
# CSV_PATH = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/list/final_dataset_factory.csv"  

# # 2. 图片源文件夹列表 (优先级：先找第一个，找不到找第二个)
# SOURCE_IMAGES_DIRS = [
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/factory/images_5k_s4_crop14",
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/factory/images_8k_s8_crop14"
# ]

# # 3. 深度图源文件夹列表 (.npy)
# SOURCE_DEPTH_DIRS = [
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/factory/output_fusion_final-5k",
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/factory/output_fusion_final-8k"
# ]

# # 4. 输出的总目录 (脚本会自动创建 images 和 depths 子文件夹)
# OUTPUT_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/Cleaned_Dataset_Factory"

# # ============================================

# def find_file_in_dirs(filename, search_dirs):
#     """在多个目录中查找文件，返回找到的第一个完整路径"""
#     for d in search_dirs:
#         full_path = os.path.join(d, filename)
#         if os.path.exists(full_path):
#             return full_path
#     return None

# def organize_files():
#     print(f"🚀 开始整理数据集...")
#     print(f"📄 读取列表: {CSV_PATH}")
    
#     # 1. 读取 CSV
#     try:
#         df = pd.read_csv(CSV_PATH)
#         # 确保列名没有空格
#         df.columns = df.columns.str.strip()
        
#         # 关键列名校验 (根据你上一轮提供的表头)
#         target_col = '匹配到的参考图(Target)'
#         if target_col not in df.columns:
#             print(f"❌ CSV 中找不到列名 '{target_col}'，请检查表头。")
#             print(f"   现有列名: {df.columns.tolist()}")
#             return
            
#         hash_list = df[target_col].dropna().unique().tolist()
#         print(f"✅ 列表中共有 {len(hash_list)} 个唯一目标文件需要提取。")
        
#     except Exception as e:
#         print(f"❌ 读取 CSV 失败: {e}")
#         return

#     # 2. 创建输出目录
#     out_img_dir = os.path.join(OUTPUT_ROOT, "images")
#     out_depth_dir = os.path.join(OUTPUT_ROOT, "depths")
    
#     os.makedirs(out_img_dir, exist_ok=True)
#     os.makedirs(out_depth_dir, exist_ok=True)
#     print(f"📂 输出目录已创建: {OUTPUT_ROOT}")

#     success_count = 0
#     missing_img_count = 0
#     missing_depth_count = 0

#     # 3. 开始遍历与复制
#     for hash_name in tqdm(hash_list, desc="Processing"):
#         # --- A. 处理图片 ---
#         src_img_path = find_file_in_dirs(hash_name, SOURCE_IMAGES_DIRS)
        
#         if src_img_path:
#             # 复制图片
#             dst_img_path = os.path.join(out_img_dir, hash_name)
#             if not os.path.exists(dst_img_path): # 避免重复复制
#                 shutil.copy2(src_img_path, dst_img_path)
#         else:
#             missing_img_count += 1
#             # print(f"⚠️ 图片丢失: {hash_name}") # 调试时可打开

#         # --- B. 处理深度图 ---
#         # 深度图可能有两种命名习惯:
#         # 1. hash.JPG -> hash.npy
#         # 2. hash.JPG -> hash.JPG.npy
        
#         base_name = os.path.splitext(hash_name)[0]
#         possible_depth_names = [
#             base_name + ".npy",      # 常见: abc.npy
#             hash_name + ".npy"       # 常见: abc.JPG.npy
#         ]
        
#         src_depth_path = None
#         found_depth_name = None
        
#         # 尝试查找这两种命名
#         for d_name in possible_depth_names:
#             found = find_file_in_dirs(d_name, SOURCE_DEPTH_DIRS)
#             if found:
#                 src_depth_path = found
#                 found_depth_name = d_name
#                 break
        
#         if src_depth_path:
#             # 复制深度图
#             # 为了规范，输出时统一命名为 hash.npy (去掉JPG后缀)
#             # 也可以保持原样，这里建议统一为 hash.npy 方便后续训练
#             final_depth_name = base_name + ".npy" 
#             dst_depth_path = os.path.join(out_depth_dir, final_depth_name)
            
#             if not os.path.exists(dst_depth_path):
#                 shutil.copy2(src_depth_path, dst_depth_path)
#         else:
#             missing_depth_count += 1
        
#         if src_img_path and src_depth_path:
#             success_count += 1

#     # 4. 总结
#     print(f"\n✨ 整理完成!")
#     print(f"   ✅ 成功归档(图+深): {success_count} 组")
#     print(f"   ⚠️ 图片缺失: {missing_img_count}")
#     print(f"   ⚠️ 深度缺失: {missing_depth_count}")
#     print(f"   📂 结果保存在: {OUTPUT_ROOT}")

# if __name__ == "__main__":
#     organize_files()

import os
import shutil
import pandas as pd
from tqdm import tqdm

# ================= ⚙️ 配置区 =================

# # 1. 你的合并后的 CSV 路径
# CSV_PATH = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/list/final_dataset_campus.csv"  

# # 2. 图片源文件夹列表 (无需担心 jpg/JPG 问题，脚本会自动处理)
# SOURCE_IMAGES_DIRS = [
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/campus",
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/campus2"
# ]

# # 3. 深度图源文件夹列表
# SOURCE_DEPTH_DIRS = [
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/output_fusion_final2-5k/depth_npy",
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/output_fusion_final2-8k/depth_npy"
# ]

# # 4. 输出目录
# OUTPUT_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/Cleaned_Dataset_Campus"
# 1. 你的合并后的 CSV 路径 (请修改为你实际的 CSV 路径)
# CSV_PATH = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/list/final_dataset_grass.csv"  

# # 2. 图片源文件夹列表 (优先级：先找第一个，找不到找第二个)
# SOURCE_IMAGES_DIRS = [
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/gress/images_5k_s4_crop14",
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/gress/images_8k_s8_crop14"
# ]

# # 3. 深度图源文件夹列表 (.npy)
# SOURCE_DEPTH_DIRS = [
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/gress/output_fusion_final-5k/depth_npy",
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/gress/output_fusion_final-8k/depth_npy"
# ]

# # 4. 输出的总目录 (脚本会自动创建 images 和 depths 子文件夹)
# OUTPUT_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/Cleaned_Dataset_Gress"




CSV_PATH = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/list/final_dataset_farm.csv"  

# 2. 图片源文件夹列表 (优先级：先找第一个，找不到找第二个)
SOURCE_IMAGES_DIRS = [
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/farm/images_5k_s4_crop14",
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/farm/images_8k_s8_crop14"
]

# 3. 深度图源文件夹列表 (.npy)
SOURCE_DEPTH_DIRS = [
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/farm/output_fusion_final-5k/depth_npy",
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2/farm/output_fusion_final2-8k/depth_npy"
]

# 4. 输出的总目录 (脚本会自动创建 images 和 depths 子文件夹)
OUTPUT_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/DJI-self2-final/Cleaned_Dataset_Farm"
# ============================================

def smart_find_file(base_name, search_dirs, file_type="image"):
    """
    智能查找文件：
    1. 根据 base_name (不带后缀的哈希名)
    2. 尝试多种后缀组合 (jpg, JPG, png...)
    3. 返回找到的第一个 (完整路径, 实际后缀)
    """
    
    # 定义我们要尝试的所有可能的后缀组合
    if file_type == "image":
        # 优先找 jpg/JPG, 也兼容 png
        candidates_exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
        # 构造候选文件名列表: [hash.jpg, hash.JPG, ...]
        candidates = [base_name + ext for ext in candidates_exts]
        
    elif file_type == "depth":
        # 深度图可能是 hash.npy, 也可能是 hash.JPG.npy, hash.jpg.npy
        candidates = [
            base_name + ".npy",            # 最标准
            base_name + ".JPG.npy",        # 常见
            base_name + ".jpg.npy",        # 常见
            base_name + ".jpeg.npy"
        ]
    
    # 开始在所有文件夹里找这些候选名
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
            
        for fname in candidates:
            full_path = os.path.join(directory, fname)
            if os.path.exists(full_path):
                return full_path, fname
                
    return None, None

def organize_files():
    print(f"🚀 开始整理数据集 (智能后缀匹配模式)...")
    
    # 1. 读取 CSV
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
        
        target_col = '匹配到的参考图(Target)'
        if target_col not in df.columns:
            print(f"❌ CSV 错误: 找不到列 '{target_col}'")
            return
            
        # 获取哈希文件名列表 (带有后缀，如 abc.JPG)
        raw_list = df[target_col].dropna().unique().tolist()
        
        # 提取纯哈希名 (去掉后缀)，作为唯一ID
        # 比如 "abc12345.JPG" -> "abc12345"
        hash_ids = [os.path.splitext(f)[0] for f in raw_list]
        # 去重
        hash_ids = list(set(hash_ids))
        
        print(f"✅ 提取到 {len(hash_ids)} 个唯一哈希ID，准备开始搜索...")
        
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 2. 创建目录
    out_img_dir = os.path.join(OUTPUT_ROOT, "images")
    out_depth_dir = os.path.join(OUTPUT_ROOT, "depths")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)

    # 3. 遍历处理
    success_count = 0
    missing_log = []

    for hid in tqdm(hash_ids, desc="Processing"):
        
        # --- A. 找图片 (自动匹配 jpg/JPG) ---
        src_img_path, img_name_found = smart_find_file(hid, SOURCE_IMAGES_DIRS, "image")
        
        # --- B. 找深度图 (自动匹配 npy/JPG.npy) ---
        src_depth_path, depth_name_found = smart_find_file(hid, SOURCE_DEPTH_DIRS, "depth")
        
        # --- C. 复制与重命名 ---
        if src_img_path and src_depth_path:
            # 1. 复制图片
            # 为了规范，我们统一输出为 .jpg (如果原图是png可能要注意，但这里假设都是jpg)
            # 这里我们使用找到的原始扩展名，或者你可以强制改成 .jpg
            dst_img_name = hid + os.path.splitext(img_name_found)[1] # 保持原后缀 (如 .JPG)
            # 或者强制小写: dst_img_name = hid + ".jpg" 
            
            shutil.copy2(src_img_path, os.path.join(out_img_dir, dst_img_name))
            
            # 2. 复制深度图
            # 统一命名为 hash.npy
            dst_depth_name = hid + ".npy"
            shutil.copy2(src_depth_path, os.path.join(out_depth_dir, dst_depth_name))
            
            success_count += 1
        else:
            # 记录缺失情况
            status = []
            if not src_img_path: status.append("缺图")
            if not src_depth_path: status.append("缺深度")
            missing_log.append(f"{hid}: {', '.join(status)}")

    # 4. 总结
    print(f"\n✨ 整理完成!")
    print(f"   ✅ 成功归档: {success_count} 组")
    print(f"   ⚠️ 数据缺失: {len(missing_log)} 组")
    
    if len(missing_log) > 0:
        print(f"   📋 缺失详情 (前10条):")
        for log in missing_log[:10]:
            print(f"      - {log}")
            
    print(f"   📂 结果保存在: {OUTPUT_ROOT}")

if __name__ == "__main__":
    organize_files()