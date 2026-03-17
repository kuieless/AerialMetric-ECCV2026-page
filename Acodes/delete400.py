# import os
# import numpy as np
# import glob
# from tqdm import tqdm

# # --- 配置参数 ---
# # 新的场景根目录 (对应 /GES-code-list/GT/)
# BASE_DIR = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES-code-list/GT" 
# # 深度图子文件夹的名称
# DEPTH_SUBFOLDER = 'npy'
# # 要设置阈值的上限
# THRESHOLD = 400
# # 替换后的值
# REPLACE_VALUE = 0
# # --- 配置参数结束 ---


# def clean_npy_files_new_path(base_dir, depth_subfolder, threshold, replace_value):
#     """
#     遍历新的场景根目录下的所有子场景，在每个场景的 'npy' 子文件夹中
#     处理所有 .npy 文件，将大于阈值的值替换为指定值，并覆盖保存。
#     """
    
#     # 查找 base_dir 下的所有场景文件夹
#     # 假设所有场景文件夹都是 base_dir 下的一级目录
#     scene_folders = [
#         d for d in os.listdir(base_dir) 
#         if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')
#     ]
    
#     if not scene_folders:
#         print(f"⚠️ 未在 '{base_dir}' 中找到任何场景文件夹。")
#         return

#     print(f"🔍 找到 {len(scene_folders)} 个场景文件夹，开始处理...")
#     print("-" * 40)

#     # 遍历每个场景文件夹 (如 dj, dj2, xg1...)
#     for scene_name in scene_folders:
#         scene_path = os.path.join(base_dir, scene_name)
#         # 深度图文件夹路径 (如 .../dj/npy)
#         depth_folder = os.path.join(scene_path, depth_subfolder)
        
#         print(f"Processing scene: {scene_name}")
        
#         if not os.path.isdir(depth_folder):
#             print(f"  -> 场景 '{scene_name}' 中未找到深度图子文件夹 '{depth_subfolder}'，跳过。")
#             continue

#         # 查找深度图文件夹内的所有 .npy 文件
#         npy_files = glob.glob(os.path.join(depth_folder, '*.npy'))
        
#         if not npy_files:
#             print(f"  -> 文件夹 '{depth_folder}' 中没有找到 .npy 文件，跳过。")
#             continue

#         # 使用 tqdm 显示处理进度
#         for file_path in tqdm(npy_files, desc=f"  清理 {scene_name}", leave=False):
#             try:
#                 # 1. 加载 .npy 文件
#                 data = np.load(file_path)
                
#                 # 2. 查找大于阈值的位置 (布尔掩码)
#                 mask = data > threshold
                
#                 # 3. 统计有多少值被修改
#                 count_modified = np.sum(mask)
                
#                 if count_modified > 0:
#                     # 4. 设置值：将大于阈值的位置替换为 REPLACE_VALUE
#                     data[mask] = replace_value
                    
#                     # 5. 覆盖保存原始文件
#                     np.save(file_path, data)
                    
#             except Exception as e:
#                 print(f"\n❌ 处理文件 '{file_path}' 时发生错误: {e}")
        
#         print(f"✅ 场景 '{scene_name}' 处理完成。")
#         print("-" * 40)

#     print("🎉 所有文件处理完毕！")

# # 执行函数
# if __name__ == "__main__":
#     # !!! 再次提醒：本操作会覆盖原始文件。请确保你已备份数据！
#     print("!!! 警告：此脚本将直接修改并覆盖您的原始 .npy 文件。请务必确认已备份数据。")
#     input("按 Enter 键继续执行，或按 Ctrl+C 取消...")
#     clean_npy_files_new_path(BASE_DIR, DEPTH_SUBFOLDER, THRESHOLD, REPLACE_VALUE)



import os
import numpy as np
import glob
from tqdm import tqdm

# ================= 配置区域 =================

# 阈值设置
THRESHOLD = 1000
REPLACE_VALUE = 0

# 文件夹候选名优先级
DEPTH_CANDIDATES = ["npy_downsampled", "npy", "depth", "depth_npy"]
# (虽然脚本只处理npy，但为了完整性保留图片候选列表定义)
IMG_CANDIDATES = ["images_downsampled", "images", "img", "rgb"]

# 请在此处粘贴你完整的 SCENE_CONFIGS 列表
SCENE_CONFIGS = [
    # ... 把你那几十个场景的字典粘贴在下面 ...
#     # 示例:
# {
#             "name": "hav",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/hav",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "train"
#         },
#         {
#             "name": "lfls",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/lfls",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "train"
#         },
#         {
#             "name": "lfls2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/lfls2",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "train"
#         },
#         {
#             "name": "lower",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/lower",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "val"
#         },
#         {
#             "name": "SMBU",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/SMBU",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "train"
#         },
#         {
#             "name": "sziit",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/sziit",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "val"
#         },
#         {
#             "name": "upper",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/upper",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "train"
#         },
#         {
#             "name": "sztu",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/sztu",
#             "intrinsics": [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726],
#             "split": "train"
#         },
# #==============================================================================

# {
#             "name": "D1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2366-down/D1",
#             "intrinsics": [846, 846, 590.65, 394.3],
#             "split": "val"
#         },
#         {
#             "name": "R1-PHD",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2366-down/R1-PHD",
#             "intrinsics": [846, 846, 590.65, 394.3],
#             "split": "val"
#         },
#         {
#             "name": "S1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2366-down/S1",
#             "intrinsics": [846, 846, 590.65, 394.3],
#             "split": "train"
#         },
#         {
#             "name": "T1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2366-down/T1",
#             "intrinsics": [846, 846, 590.65, 394.3],
#             "split": "train"
#         },
#         {
#             "name": "T2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2366-down/T2",
#             "intrinsics": [846, 846, 590.65, 394.3],
#             "split": "train"
#         },
#         {
#             "name": "R1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/R1",
#             "intrinsics": [876.77532958984375, 876.77532958984375, 690.36, 457.3],
#             "split": "val"
#         },

# #===================================================================================
#         {
#             "name": "BC1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2044-down/BC1",
#             "intrinsics": [982.5, 982.5, 512, 342.8],
#             "split": "train"
#         },
#         {
#             "name": "BC2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2044-down/BC2",
#             "intrinsics": [982.5, 982.5, 512, 342.8],
#             "split": "train"
#         },
#         {
#             "name": "L1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ESRI/2044-down/L1",
#             "intrinsics": [982.5, 982.5, 512, 342.8],
#             "split": "train"
#         },
# #==================================================================================================
# {
#             "name": "dj",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "dj2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj2",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "dj3",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj3",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "val"
#         },
#         {
#             "name": "dj4",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj4",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "hsd1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/hsd1",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "val"
#         },
#         {
#             "name": "lm",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/lm",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "xg1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg1",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "xg2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg2",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "xg3",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg3",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "xg4",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg4",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "train"
#         },
#         {
#             "name": "xg5",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg5",
#             "intrinsics": [467.79, 467.79, 479.6, 269.5],
#             "split": "val"
#         },

# #=======================================================================================================================
# {
#             "name": "SYS-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/1",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-10",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/10",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-11",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/11",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-13",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/13",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-14",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/14",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-16",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/16",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-17",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/17",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/2",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-3",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/3",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-4",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/4",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-5",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/5",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-6",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/6",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-7",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/7",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-8",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/8",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },
#         {
#             "name": "SYS-9",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/SYS-down/9",
#             "intrinsics": [1012.6, 1012.6, 503.5, 345.3],
#             "split": "train"
#         },

# #======================================================================================================================================
#         {
#             "name": "ainterval5_AMtown01",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/ainterval5_AMtown01_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "val"
#         },
#         {
#             "name": "interval5_AMvalley01",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_AMvalley01_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKairport02",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport02_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKairport_GNSS03",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport_GNSS03_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKisland_GNSS01",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKisland_GNSS01_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "ainterval5_AMtown02",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/ainterval5_AMtown02_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "val"
#         },
#         {
#             "name": "interval5_AMvalley02",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_AMvalley02_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKairport03",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport03_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKairport_GNSS_Evening",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport_GNSS_Evening_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKisland_GNSS02",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKisland_GNSS02_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "ainterval5_HKisland02",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/ainterval5_HKisland02_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_AMvalley03",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_AMvalley03_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKairport_GNSS01",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport_GNSS01_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKisland01",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKisland01_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKisland_GNSS03",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKisland_GNSS03_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_AMtown03",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_AMtown03_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "val"
#         },
#         {
#             "name": "interval5_HKairport01",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport01_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKairport_GNSS02",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKairport_GNSS02_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKisland03",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKisland03_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },
#         {
#             "name": "interval5_HKisland_GNSS_Evening",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UAV/UAV_downsampled_1008/interval5_HKisland_GNSS_Evening_cropped_downsampled",
#             "intrinsics": [534.7, 534.7, 504, 504],
#             "split": "train"
#         },



# #==============================================================================================
#         {
#             "name": "yuehai10-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai10-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai11-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai11-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai13-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai13-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai14-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai14-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai15-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai15-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai16-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai16-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai2-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai2-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai5-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai5-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai6-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai6-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai7-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai7-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai8-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai8-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai9-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai9-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai10-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai10-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai12-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai12-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai13-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai13-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai15-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai15-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai16-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai16-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai17-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai17-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai4-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai4-1",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai5-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai5-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai6-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai6-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai7-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai7-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai8-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai8-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },
#         {
#             "name": "yuehai9-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/UrbanBIS-down/yuehai9-2",
#             "intrinsics": [1012.6, 1012.6, 511, 339.5],
#             "split": "train"
#         },

# {
#             "name": "park5",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park5",
#             "intrinsics": [595.81, 595.81, 501.80, 371.66],
#             "split": "train"
#         },
#         {
#             "name": "ODM8-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM8-2",
#             "intrinsics": [1047.60, 1047.60, 495.33, 376.83],
#             "split": "train"
#         },
#         {
#             "name": "aukerman",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/aukerman",
#             "intrinsics": [910.14, 910.14, 626.12, 444.92],
#             "split": "train"
#         },
#         {
#             "name": "ODM37",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM37",
#             "intrinsics": [686.47, 686.47, 483.86, 369.94],
#             "split": "train"
#         },
#         {
#             "name": "bellus",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/bellus",
#             "intrinsics": [725.12, 725.12, 510.92, 381.13],
#             "split": "train"
#         },
#         {
#             "name": "brighton-beach",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/brighton-beach",
#             "intrinsics": [650.51, 650.51, 500.32, 291.02],
#             "split": "train"
#         },
#         {
#             "name": "ODM-32",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-32",
#             "intrinsics": [609.37, 609.37, 495.02, 368.49],
#             "split": "train"
#         },
#         {
#             "name": "park13",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park13",
#             "intrinsics": [599.01, 599.01, 501.69, 371.41],
#             "split": "train"
#         },
#         {
#             "name": "park12",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park12",
#             "intrinsics": [575.43, 575.43, 502.32, 372.81],
#             "split": "train"
#         },
#         {
#             "name": "park9",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park9",
#             "intrinsics": [573.97, 573.97, 502.53, 372.67],
#             "split": "train"
#         },
#         {
#             "name": "ODM2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM2",
#             "intrinsics": [911.34, 911.34, 676.57, 460.39],
#             "split": "train"
#         },
#         {
#             "name": "ODM-27",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-27",
#             "intrinsics": [1065.84, 1065.84, 675.08, 448.64],
#             "split": "train"
#         },
#         {
#             "name": "ODM39",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM39",
#             "intrinsics": [689.75, 689.75, 483.19, 375.89],
#             "split": "train"
#         },
#         {
#             "name": "ODM-45",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-45",
#             "intrinsics": [725.28, 725.28, 510.93, 381.07],
#             "split": "train"
#         },
#         {
#             "name": "park4",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park4",
#             "intrinsics": [570.78, 570.78, 502.30, 372.30],
#             "split": "train"
#         },
#         {
#             "name": "ODM8-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM8-1",
#             "intrinsics": [1080.68, 1080.68, 496.57, 392.89],
#             "split": "train"
#         },
#         {
#             "name": "ODM-41",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-41",
#             "intrinsics": [650.51, 650.51, 500.32, 291.02],
#             "split": "train"
#         },
#         {
#             "name": "park11",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park11",
#             "intrinsics": [584.03, 584.03, 501.06, 372.34],
#             "split": "train"
#         },
#         {
#             "name": "ODM-44",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-44",
#             "intrinsics": [908.82, 908.82, 626.16, 445.03],
#             "split": "train"
#         },
#         {
#             "name": "park0",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park0",
#             "intrinsics": [758.61, 758.61, 495.53, 362.35],
#             "split": "train"
#         },
#         {
#             "name": "caliterra",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/caliterra",
#             "intrinsics": [748.93, 748.93, 513.97, 383.91],
#             "split": "train"
#         },
#         {
#             "name": "ODM-34-1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-34-1",
#             "intrinsics": [909.48, 909.48, 621.32, 445.99],
#             "split": "train"
#         },
#         {
#             "name": "park8",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park8",
#             "intrinsics": [626.77, 626.77, 501.11, 369.93],
#             "split": "train"
#         },
#         {
#             "name": "park2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park2",
#             "intrinsics": [593.51, 593.51, 501.83, 371.86],
#             "split": "train"
#         },
#         {
#             "name": "ODM1",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM1",
#             "intrinsics": [911.82, 911.82, 676.26, 460.79],
#             "split": "train"
#         },
#         {
#             "name": "park10",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park10",
#             "intrinsics": [616.13, 616.13, 499.97, 370.59],
#             "split": "train"
#         },
#         {
#             "name": "lewis",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/lewis",
#             "intrinsics": [725.25, 725.25, 491.34, 355.60],
#             "split": "train"
#         },
#         {
#             "name": "park6",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park6",
#             "intrinsics": [596.75, 596.75, 501.54, 371.70],
#             "split": "train"
#         },
#         {
#             "name": "ODM-49",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-49",
#             "intrinsics": [641.28, 641.28, 451.24, 331.58],
#             "split": "train"
#         },
#         {
#             "name": "garfield_msp",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/garfield_msp",
#             "intrinsics": [2265.05, 2265.05, 279.00, 45.72],
#             "split": "train"
#         },
#         {
#             "name": "park14",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park14",
#             "intrinsics": [613.32, 613.32, 500.69, 371.01],
#             "split": "train"
#         },
#         {
#             "name": "park3",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park3",
#             "intrinsics": [603.64, 603.64, 501.51, 371.84],
#             "split": "train"
#         },
#         {
#             "name": "ODM6",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM6",
#             "intrinsics": [597.12, 597.12, 511.00, 356.07],
#             "split": "train"
#         },
#         {
#             "name": "ODM-34-2",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-34-2",
#             "intrinsics": [907.86, 907.86, 620.98, 446.76],
#             "split": "train"
#         },
#         {
#             # 注意: 这里原始数据写的是 DOM-47, 请确认文件夹名是否为 ODM-47 还是 DOM-47
#             "name": "DOM-47",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/DOM-47",
#             "intrinsics": [1727.76, 1727.76, 219.82, 245.31],
#             "split": "train"
#         },
#         {
#             "name": "ODM-48",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-48",
#             "intrinsics": [732.09, 732.09, 491.24, 353.73],
#             "split": "train"
#         },
#         {
#             "name": "ODM-50",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-50",
#             "intrinsics": [666.82, 666.82, 483.63, 379.99],
#             "split": "train"
#         },
#         {
#             "name": "ODM3",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM3",
#             "intrinsics": [592.99, 592.99, 499.06, 375.73],
#             "split": "train"
#         },
#         {
#             "name": "seneca",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/seneca",
#             "intrinsics": [641.27, 641.27, 451.25, 331.56],
#             "split": "train"
#         },
#         {
#             "name": "park7",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/park7",
#             "intrinsics": [578.35, 578.35, 501.97, 372.50],
#             "split": "train"
#         },
#         {
#             "name": "ODM-46",
#             "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/ODM/ODM-46",
#             "intrinsics": [748.97, 748.97, 513.97, 383.91],
#             "split": "train"
#         },

                {
            "name": "yingrenshi1",
            "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/yingrenhsi1",
            "intrinsics": [920.1700439453125, 920.1700439453125, 676, 450],
            "split": "train"
        },
        {
            "name": "yingrenshi2",
            "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/yingrenshi2",
            "intrinsics": [920.1700439453125, 920.1700439453125, 676, 450],
            "split": "train"
        },

    # ... 确保你粘贴了完整的列表 ...
]

# ================= 功能函数 =================

def find_valid_folder(root_path, candidates):
    """
    在 root_path 下寻找存在的子文件夹，按照 candidates 的顺序查找。
    返回找到的第一个完整路径，如果都没找到则返回 None。
    """
    if not os.path.exists(root_path):
        return None
        
    for sub in candidates:
        full_path = os.path.join(root_path, sub)
        if os.path.isdir(full_path):
            return full_path
    return None

def batch_clean_depth_maps(configs):
    print(f"🚀 开始批量处理 {len(configs)} 个场景...")
    print(f"🎯 目标: 将 > {THRESHOLD} 的深度值替换为 {REPLACE_VALUE}")
    print("="*60)

    # 总体进度条
    total_scenes = len(configs)
    
    for idx, scene in enumerate(configs):
        name = scene['name']
        root = scene['root_dir']
        
        # 1. 自动寻找 npy 文件夹
        depth_dir = find_valid_folder(root, DEPTH_CANDIDATES)
        
        prefix = f"[{idx+1}/{total_scenes}] {name}"
        
        if not depth_dir:
            print(f"⚠️  {prefix}: 无法在 {root} 下找到任何匹配的深度文件夹 (Candidates: {DEPTH_CANDIDATES})")
            continue

        # 2. 获取所有 .npy 文件
        npy_files = glob.glob(os.path.join(depth_dir, "*.npy"))
        if not npy_files:
            print(f"⚠️  {prefix}: 文件夹存在但为空 -> {depth_dir}")
            continue

        # 3. 处理当前场景的文件
        # print(f"✅ {prefix}: 找到目录 -> {os.path.basename(depth_dir)} ({len(npy_files)} files)")
        
        modified_count = 0
        
        # 使用 tqdm 显示当前场景的处理进度，leave=False 表示处理完后清除进度条，保持界面整洁
        for f_path in tqdm(npy_files, desc=f"Processing {name}", leave=False):
            try:
                # 加载
                data = np.load(f_path)
                
                # 检查是否需要修改 (先用 mask 判断，避免无意义的写入IO)
                mask = data > THRESHOLD
                if np.any(mask):
                    # 修改数值
                    data[mask] = REPLACE_VALUE
                    # 覆盖保存
                    np.save(f_path, data)
                    modified_count += 1
                    
            except Exception as e:
                print(f"❌ {prefix}: 读取/保存文件失败 {os.path.basename(f_path)} - {e}")

        # 场景处理完毕的小结
        if modified_count > 0:
            print(f"✅ {prefix}: 已清理 {modified_count}/{len(npy_files)} 个文件 (Dir: {os.path.basename(depth_dir)})")
        else:
            print(f"🆗 {prefix}: 无需修改 (所有值均 <= {THRESHOLD})")

    print("="*60)
    print("🎉 所有场景处理完毕！")

# ================= 执行入口 =================

if __name__ == "__main__":
    if not SCENE_CONFIGS:
        print("❌ 错误：SCENE_CONFIGS 列表为空，请在代码中填入配置。")
    else:
        print("!!! 警告：此操作将永久修改原始 .npy 文件。!!!")
        user_input = input("确认执行吗? (输入 'yes' 继续): ")
        if user_input.lower() == 'yes':
            batch_clean_depth_maps(SCENE_CONFIGS)
        else:
            print("操作已取消。")