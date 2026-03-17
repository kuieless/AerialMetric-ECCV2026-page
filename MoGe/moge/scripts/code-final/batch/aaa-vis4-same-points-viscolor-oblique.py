import os
import sys
import numpy as np
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.subplots as plt_subplots
import matplotlib.pyplot as plt

# ================= 1. 用户配置区域 =================
TARGET_SIZE = (1280, 960) # (宽, 高)

# 多个根目录配置
ROOT_DIRS = [
    "/data1/szq/TrainingData_Final_MoGe-All",
    "/data1/szq/FOV",
    "/data1/szq/B-2",
    "/data1/szq/B"
]

SAMPLE_PER_SCENE = 5 # 每个场景随机抽取的数量

OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/fig1-8_sampled3" 

MANUAL_PROBES = [
    (320, 480),
    (880, 400) 
]
USE_MANUAL_PROBES = True
COLORMAP = 'Spectral'       
INVALID_COLOR = 'white'     

# 稀疏深度图判断阈值 (如果有效像素占比低于 5%，判定为稀疏雷达/点云深度)
SPARSE_THRESHOLD = 0.05 

# ================= 2. 核心渲染器 & 工具 =================

def get_valid_mask(depth_array: np.ndarray) -> np.ndarray:
    return np.isfinite(depth_array) & (depth_array > 1e-3) & (depth_array < 1000.0)

def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
    """
    如果目标点有效，直接返回；否则寻找欧式距离最近的有效点。
    """
    if valid_mask[tgt_y, tgt_x]: 
        return (tgt_x, tgt_y)
    
    ys, xs = np.where(valid_mask)
    if len(ys) == 0: 
        return None 
    
    # 计算所有有效点到目标点的距离平方
    dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
    best_idx = np.argmin(dist_sq)
    return (int(xs[best_idx]), int(ys[best_idx]))

class DepthRenderer:
    def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
        self.cmap = plt.get_cmap(cmap_name).copy()
        self.cmap.set_bad(color=invalid_color) 

    def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, target_probes: list):
        mask = get_valid_mask(depth_array)
        if not np.any(mask): return
            
        valid_ratio = np.sum(mask) / mask.size
        is_sparse = valid_ratio < SPARSE_THRESHOLD

        # ================= 针对稀疏深度的特殊处理 =================
        if is_sparse:
            # 1. 对有效深度点进行膨胀处理，让点变大
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            depth_clean = np.where(mask, depth_array, 0.0)
            depth_array_vis = cv2.dilate(depth_clean, kernel)
            mask_vis = depth_array_vis > 1e-3
            
            # 2. 稀疏图 Resize 使用最近邻插值，保证点不被抹匀
            depth_resized = cv2.resize(depth_array_vis, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            mask_resized = cv2.resize(mask_vis.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        else:
            # 稠密图使用双线性插值
            depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        
        # ================= 2% - 98% 局部自适应色阶 =================
        valid_pixels = depth_resized[mask_resized]
        if len(valid_pixels) == 0: return
        
        local_vmin = np.percentile(valid_pixels, 2)
        local_vmax = np.percentile(valid_pixels, 98)
        
        if local_vmax <= local_vmin:
            local_vmax = local_vmin + 1e-3
            
        depth_clipped = np.clip(depth_resized, local_vmin, local_vmax)
        depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        
        path_pure = out_dir / f"{stem}_pure.png"
        path_probes = out_dir / f"{stem}_probes.png"

        # 使用防抖画布
        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        ax.set_xlim(0, TARGET_SIZE[0])
        ax.set_ylim(TARGET_SIZE[1], 0)
        ax.axis('off')

        ax.imshow(depth_masked, cmap=self.cmap, vmin=local_vmin, vmax=local_vmax, 
                  extent=[0, TARGET_SIZE[0], TARGET_SIZE[1], 0])

        fig.savefig(path_pure, pad_inches=0)

        # --- 画探针 ---
        orig_h, orig_w = depth_array.shape
        scale_x = TARGET_SIZE[0] / orig_w
        scale_y = TARGET_SIZE[1] / orig_h

        colors = ['#FF3366', '#00E676'] 
        for pt_idx, (orig_px, orig_py) in enumerate(target_probes):
            # 将探针坐标从原图尺寸映射到 TARGET_SIZE 画布上
            new_x, new_y = int(orig_px * scale_x), int(orig_py * scale_y)
            c = colors[pt_idx % len(colors)]
            
            ax.plot(new_x, new_y, marker='o', markerfacecolor=c, markeredgecolor='white', 
                    markersize=12, markeredgewidth=2, zorder=5)
            
            # 【重要修改】：直接从原图矩阵 depth_array 中获取真实有效数值，而不是被缩放或膨胀过的值
            val = depth_array[orig_py, orig_px]
            
            text_str = f"{val:.1f}m"
            y_offset = -40 if pt_idx == 0 else 40 
            ax.text(new_x, new_y + y_offset, text_str, color='black', fontsize=14, fontweight='bold',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor="none", alpha=0.8))

        fig.savefig(path_probes, pad_inches=0)
        plt.close(fig)

# ================= 3. 数据扫描与加载 =================

def load_depth_data(file_path: Path):
    try:
        if file_path.suffix.lower() == '.npy':
            data = np.load(file_path)
            if data.ndim == 3 and data.shape[0] == 1: data = data[0]
            elif data.ndim == 3 and data.shape[2] == 1: data = data[:, :, 0]
            if data.ndim == 3 and data.shape[2] == 3: return None
            return data.astype(np.float32)
        elif file_path.suffix.lower() == '.png':
            data = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
            if data is None: return None
            return data.astype(np.float32) / 1000.0
    except Exception as e:
        print(f"❌ 读取异常 {file_path}: {e}")
    return None

def gather_sample_tasks(root_dirs, samples_per_scene):
    tasks = []
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            print(f"⚠️ 根目录不存在，跳过: {root}")
            continue
            
        dataset_name = root_path.name
        
        # 遍历根目录下的所有场景目录 (例如 BC1)
        for scene_dir in root_path.iterdir():
            if not scene_dir.is_dir(): continue
            
            scene_name = scene_dir.name
            frame_depth_paths = []
            
            # 遍历场景目录下的所有帧目录 (例如 000000)
            for frame_dir in scene_dir.iterdir():
                if not frame_dir.is_dir(): continue
                
                depth_file = frame_dir / "depth.npy"
                if depth_file.exists():
                    frame_depth_paths.append(depth_file)
            
            # 随机抽样
            if frame_depth_paths:
                sampled_paths = random.sample(frame_depth_paths, min(samples_per_scene, len(frame_depth_paths)))
                for path in sampled_paths:
                    frame_name = path.parent.name
                    unique_name = f"{dataset_name}_{scene_name}_{frame_name}"
                    tasks.append((unique_name, path))
                    
    return tasks

if __name__ == "__main__":
    print(f"{'='*60}\n📊 多场景随机采样 & 原图拷贝 & 自适应渲染器\n{'='*60}")
    
    # 1. 扫描文件并采样
    print("🔍 正在扫描文件树并进行随机抽样...")
    sample_tasks = gather_sample_tasks(ROOT_DIRS, SAMPLE_PER_SCENE)
    
    if not sample_tasks:
        print("❌ 未在指定的目录中找到任何深度数据。")
        sys.exit(1)
        
    print(f"✅ 扫描完毕，共抽取了 {len(sample_tasks)} 个样本准备处理。")
    
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = DepthRenderer()
    
    # 2. 逐图加载与渲染
    for unique_name, file_path in tqdm(sample_tasks, desc="Processing"):
        data = load_depth_data(file_path)
        if data is None: 
            continue
            
        mask = get_valid_mask(data)
        if not np.any(mask):
            continue
        
        # ================== 寻找并拷贝原图 ==================
        # 优先找 jpg，找不到找 png
        img_src_path = file_path.parent / "image.jpg"
        if not img_src_path.exists():
            img_src_path = file_path.parent / "image.png"
            
        if img_src_path.exists():
            img_dst_path = out_dir / f"{unique_name}_image{img_src_path.suffix}"
            try:
                shutil.copy2(img_src_path, img_dst_path)
            except Exception as e:
                print(f"⚠️ 拷贝原图失败 {img_src_path}: {e}")
            
        # ================== 探针坐标换算与吸附 ==================
        h, w = data.shape
        local_probes = []
        if USE_MANUAL_PROBES and MANUAL_PROBES:
            for px, py in MANUAL_PROBES:
                # 按照 TARGET_SIZE 反算回原图尺寸比例
                px_scaled = int(px * (w / TARGET_SIZE[0]))
                py_scaled = int(py * (h / TARGET_SIZE[1]))
                px_scaled = max(0, min(px_scaled, w - 1))
                py_scaled = max(0, min(py_scaled, h - 1))
                
                # 吸附到最近的有效点 (核心逻辑：容错处理)
                pt = get_nearest_valid_point(px_scaled, py_scaled, mask)
                if pt is not None: 
                    local_probes.append(pt)
        else:
            cx, cy = w // 2, h // 2
            offset_x = w // 4
            pt1 = get_nearest_valid_point(cx - offset_x, cy, mask)
            pt2 = get_nearest_valid_point(cx + offset_x, cy, mask)
            local_probes = [pt for pt in [pt1, pt2] if pt is not None]

        # ================== 渲染出图 ==================
        renderer.render_trio(
            depth_array=data,
            stem=unique_name,
            out_dir=out_dir,
            target_probes=local_probes
        )
            
    print(f"\n✅ 任务完成！RGB图与深度图已保存至: {out_dir}")
# import os
# import sys
# import numpy as np
# import cv2
# import random
# from pathlib import Path
# from tqdm import tqdm
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # ================= 1. 用户配置区域 =================
# TARGET_SIZE = (1280, 960) # (宽, 高)

# # 多个根目录配置
# ROOT_DIRS = [
#     "/data1/szq/TrainingData_Final_MoGe-All",
#     "/data1/szq/FOV",
#     "/data1/szq/B-2",
#     "/data1/szq/B"
# ]

# SAMPLE_PER_SCENE = 5 # 每个场景随机抽取的数量

# OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/fig1-8_sampled2" 

# MANUAL_PROBES = [
#     (320, 480),
#     (880, 400) 
# ]
# USE_MANUAL_PROBES = True
# COLORMAP = 'Spectral'       
# INVALID_COLOR = 'white'     

# # 稀疏深度图判断阈值 (如果有效像素占比低于 5%，判定为稀疏雷达/点云深度)
# SPARSE_THRESHOLD = 0.05 

# # ================= 2. 核心渲染器 & 工具 =================

# def get_valid_mask(depth_array: np.ndarray) -> np.ndarray:
#     return np.isfinite(depth_array) & (depth_array > 1e-3) & (depth_array < 1000.0)

# def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
#     if valid_mask[tgt_y, tgt_x]: return (tgt_x, tgt_y)
#     ys, xs = np.where(valid_mask)
#     if len(ys) == 0: return None 
#     dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
#     best_idx = np.argmin(dist_sq)
#     return (int(xs[best_idx]), int(ys[best_idx]))

# class DepthRenderer:
#     def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
#         self.cmap = plt.get_cmap(cmap_name).copy()
#         self.cmap.set_bad(color=invalid_color) 

#     def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, target_probes: list):
#         mask = get_valid_mask(depth_array)
#         if not np.any(mask): return
            
#         valid_ratio = np.sum(mask) / mask.size
#         is_sparse = valid_ratio < SPARSE_THRESHOLD

#         # ================= 针对稀疏深度的特殊处理 =================
#         if is_sparse:
#             # 1. 对有效深度点进行膨胀处理
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#             depth_clean = np.where(mask, depth_array, 0.0)
#             depth_array = cv2.dilate(depth_clean, kernel)
#             mask = depth_array > 1e-3
            
#             # 2. 稀疏图 Resize 使用最近邻插值
#             depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
#             mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
#         else:
#             # 稠密图使用双线性插值
#             depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
#             mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        
#         # ================= 核心修改：2% - 98% 局部自适应色阶 =================
#         valid_pixels = depth_resized[mask_resized]
#         if len(valid_pixels) == 0: return
        
#         # 提取 2% 和 98% 的分位数，过滤掉极端的近景噪点和远景噪点
#         local_vmin = np.percentile(valid_pixels, 2)
#         local_vmax = np.percentile(valid_pixels, 98)
        
#         # 避免遇到单一深度值（如纯平墙面）导致 vmin == vmax 报错
#         if local_vmax <= local_vmin:
#             local_vmax = local_vmin + 1e-3
            
#         # 截断超出范围的值，并将无效区域设为 NaN 以便填充 invalid_color (白色)
#         depth_clipped = np.clip(depth_resized, local_vmin, local_vmax)
#         depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        
#         path_pure = out_dir / f"{stem}_pure.png"
#         path_probes = out_dir / f"{stem}_probes.png"

#         # 使用防抖画布
#         fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
#         fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
#         ax.set_xlim(0, TARGET_SIZE[0])
#         ax.set_ylim(TARGET_SIZE[1], 0)
#         ax.axis('off')

#         # 传入计算好的 local_vmin 和 local_vmax
#         ax.imshow(depth_masked, cmap=self.cmap, vmin=local_vmin, vmax=local_vmax, 
#                   extent=[0, TARGET_SIZE[0], TARGET_SIZE[1], 0])

#         fig.savefig(path_pure, pad_inches=0)

#         # --- 画探针 ---
#         orig_h, orig_w = depth_array.shape
#         scale_x = TARGET_SIZE[0] / orig_w
#         scale_y = TARGET_SIZE[1] / orig_h

#         colors = ['#FF3366', '#00E676'] 
#         for pt_idx, (px, py) in enumerate(target_probes):
#             new_x, new_y = int(px * scale_x), int(py * scale_y)
#             c = colors[pt_idx % len(colors)]
            
#             ax.plot(new_x, new_y, marker='o', markerfacecolor=c, markeredgecolor='white', 
#                     markersize=12, markeredgewidth=2, zorder=5)
            
#             val = depth_resized[new_y, new_x]
#             text_str = f"{val:.1f}m"
#             y_offset = -40 if pt_idx == 0 else 40 
#             ax.text(new_x, new_y + y_offset, text_str, color='black', fontsize=14, fontweight='bold',
#                     ha='center', va='center', zorder=6,
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor="none", alpha=0.8))

#         fig.savefig(path_probes, pad_inches=0)
#         plt.close(fig)

# # ================= 3. 数据扫描与加载 =================

# def load_depth_data(file_path: Path):
#     try:
#         if file_path.suffix.lower() == '.npy':
#             data = np.load(file_path)
#             if data.ndim == 3 and data.shape[0] == 1: data = data[0]
#             elif data.ndim == 3 and data.shape[2] == 1: data = data[:, :, 0]
#             if data.ndim == 3 and data.shape[2] == 3: return None
#             return data.astype(np.float32)
#         elif file_path.suffix.lower() == '.png':
#             data = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
#             if data is None: return None
#             return data.astype(np.float32) / 1000.0
#     except Exception as e:
#         print(f"❌ 读取异常 {file_path}: {e}")
#     return None

# def gather_sample_tasks(root_dirs, samples_per_scene):
#     tasks = []
#     for root in root_dirs:
#         root_path = Path(root)
#         if not root_path.exists():
#             print(f"⚠️ 根目录不存在，跳过: {root}")
#             continue
            
#         dataset_name = root_path.name
        
#         # 遍历根目录下的所有场景目录 (例如 BC1)
#         for scene_dir in root_path.iterdir():
#             if not scene_dir.is_dir(): continue
            
#             scene_name = scene_dir.name
#             frame_depth_paths = []
            
#             # 遍历场景目录下的所有帧目录 (例如 000000)
#             for frame_dir in scene_dir.iterdir():
#                 if not frame_dir.is_dir(): continue
                
#                 depth_file = frame_dir / "depth.npy"
#                 if depth_file.exists():
#                     frame_depth_paths.append(depth_file)
            
#             # 随机抽样
#             if frame_depth_paths:
#                 sampled_paths = random.sample(frame_depth_paths, min(samples_per_scene, len(frame_depth_paths)))
#                 for path in sampled_paths:
#                     # 将来源信息带上，方便后续重命名输出图片
#                     frame_name = path.parent.name
#                     unique_name = f"{dataset_name}_{scene_name}_{frame_name}"
#                     tasks.append((unique_name, path))
                    
#     return tasks

# if __name__ == "__main__":
#     print(f"{'='*60}\n📊 多场景随机采样 & 稀疏自适应渲染器\n{'='*60}")
    
#     # 1. 扫描文件并采样
#     print("🔍 正在扫描文件树并进行随机抽样...")
#     sample_tasks = gather_sample_tasks(ROOT_DIRS, SAMPLE_PER_SCENE)
    
#     if not sample_tasks:
#         print("❌ 未在指定的目录中找到任何深度数据。")
#         sys.exit(1)
        
#     print(f"✅ 扫描完毕，共抽取了 {len(sample_tasks)} 张深度图进行渲染。")
    
#     out_dir = Path(OUTPUT_DIR)
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     renderer = DepthRenderer()
    
#     # 2. 逐图加载与渲染
#     for unique_name, file_path in tqdm(sample_tasks, desc="Rendering"):
#         data = load_depth_data(file_path)
#         if data is None: 
#             continue
            
#         mask = get_valid_mask(data)
#         if not np.any(mask):
#             continue
            
#         # 计算探针位置 (如果是随机单图，每张图独立寻找合法探针)
#         h, w = data.shape
#         local_probes = []
#         if USE_MANUAL_PROBES and MANUAL_PROBES:
#             for px, py in MANUAL_PROBES:
#                 # 将探针映射到当前图的比例上
#                 px_scaled = int(px * (w / 1280))
#                 py_scaled = int(py * (h / 960))
#                 px_scaled = max(0, min(px_scaled, w - 1))
#                 py_scaled = max(0, min(py_scaled, h - 1))
                
#                 pt = get_nearest_valid_point(px_scaled, py_scaled, mask)
#                 if pt is not None: 
#                     local_probes.append(pt)
#         else:
#             cx, cy = w // 2, h // 2
#             offset_x = w // 4
#             pt1 = get_nearest_valid_point(cx - offset_x, cy, mask)
#             pt2 = get_nearest_valid_point(cx + offset_x, cy, mask)
#             local_probes = [pt for pt in [pt1, pt2] if pt is not None]

#         # 渲染出图
#         renderer.render_trio(
#             depth_array=data,
#             stem=unique_name,
#             out_dir=out_dir,
#             target_probes=local_probes
#         )
            
#     print(f"\n✅ 全部渲染完成！输出目录: {out_dir}")

# # import os
# # import sys
# # import numpy as np
# # import cv2
# # from pathlib import Path
# # from tqdm import tqdm
# # import matplotlib
# # matplotlib.use('Agg')
# # # import matplotlib.subplots as plt_subplots
# # import matplotlib.pyplot as plt

# # # ================= 1. 用户配置区域 =================
# # TARGET_SIZE = (1280, 960) # (宽, 高)
# # INPUT_TARGETS = [
# # "/data1/szq/Val/Viss/Vis-sota/pic-npy/000025.npy", 
# # "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/000025.npy", 
# # "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/pro000025.npy", 
# # "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/000025.npy", 
# # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/000025/depth.npy", 
# # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/000025/depth.npy", 

# # # "/data1/szq/Val/Viss/Vis-sota/pic-npy/000099.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/000099.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/000099.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/000099.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/000099/depth.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/000099/depth.npy", 

# # # "/data1/szq/Val/Viss/Vis-sota/pic-npy/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b/depth.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b/depth.npy", 


# # # "/data1/szq/Val/Viss/Vis-sota/pic-npy/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/1ddfb001659b9cfe2e10a82b66ec1c065901973f/depth.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/1ddfb001659b9cfe2e10a82b66ec1c065901973f/depth.npy", 


# # # "/data1/yzy/Benchmark/Metric3D/data/nyu_demo/depth/sync_depth_00000.png", 
# # # "/data1/szq/Val/Viss/Vis-sota/FInal/ground/zoeimage.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/FInal/ground/proimage.npy", 
# # # "/data1/szq/Val/Viss/Vis-sota/FInal/ground/image.npy", 
# # # "/data1/szq/Val/Viss/vissss/nyu/Baseline/rgb_00000/depth.npy", 
# # # "/data1/szq/Val/Viss/vissss/nyu/LoRA_Rank96/rgb_00000/loradepth.npy", 



# # ]

# # OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/fig1-8" 
# # MANUAL_PROBES = [
# #     (320, 480),
# #     (880, 400) 
# # ]
# # USE_MANUAL_PROBES = True
# # COLORMAP = 'Spectral'       
# # INVALID_COLOR = 'white'     

# # # ================= 2. 核心渲染器 & 工具 =================

# # def get_valid_mask(depth_array: np.ndarray) -> np.ndarray:
# #     return np.isfinite(depth_array) & (depth_array > 1e-3) & (depth_array < 1000.0)

# # def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
# #     if valid_mask[tgt_y, tgt_x]: return (tgt_x, tgt_y)
# #     ys, xs = np.where(valid_mask)
# #     if len(ys) == 0: return None 
# #     dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
# #     best_idx = np.argmin(dist_sq)
# #     return (int(xs[best_idx]), int(ys[best_idx]))


# # class DepthRenderer:
# #     def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
# #         self.cmap = plt.get_cmap(cmap_name).copy()
# #         self.cmap.set_bad(color=invalid_color) 

# #     def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, target_probes: list, global_vmax: float):
# #         mask = get_valid_mask(depth_array)
# #         if not np.any(mask): return
            
# #         depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
# #         mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        
# #         # 1. 获取当前图的自身量程 (99% 避免噪点)
# #         valid_pixels = depth_resized[mask_resized]
# #         local_vmax = np.percentile(valid_pixels, 99)
        
# #         # 2. 核心魔法：计算动态带宽
# #         ratio = min(local_vmax / global_vmax, 1.0)
# #         # 强制保底 30% 的色带宽度，确保有细节 (可按喜好微调 0.3)
# #         band = 0.3 + 0.7 * ratio 
        
# #         # 3. 反推 matplotlib 需要的 vmax
# #         plot_vmax = local_vmax / band
        
# #         depth_clipped = np.clip(depth_resized, 0, plot_vmax)
# #         depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        
# #         path_pure = out_dir / f"{stem}_pure.png"
# #         path_probes = out_dir / f"{stem}_probes.png"

# #         # 使用防抖画布
# #         fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
# #         fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
# #         ax.set_xlim(0, TARGET_SIZE[0])
# #         ax.set_ylim(TARGET_SIZE[1], 0)
# #         ax.axis('off')

# #         # 统一以 0 为起点，plot_vmax 为终点
# #         ax.imshow(depth_masked, cmap=self.cmap, vmin=0, vmax=plot_vmax, 
# #                   extent=[0, TARGET_SIZE[0], TARGET_SIZE[1], 0])

# #         fig.savefig(path_pure, pad_inches=0)

# #         # --- 画探针 ---
# #         orig_h, orig_w = depth_array.shape
# #         scale_x = TARGET_SIZE[0] / orig_w
# #         scale_y = TARGET_SIZE[1] / orig_h

# #         colors = ['#FF3366', '#00E676'] 
# #         for pt_idx, (px, py) in enumerate(target_probes):
# #             new_x, new_y = int(px * scale_x), int(py * scale_y)
# #             c = colors[pt_idx % len(colors)]
            
# #             ax.plot(new_x, new_y, marker='o', markerfacecolor=c, markeredgecolor='white', 
# #                     markersize=12, markeredgewidth=2, zorder=5)
            
# #             val = depth_resized[new_y, new_x]
# #             text_str = f"{val:.1f}m"
# #             y_offset = -40 if pt_idx == 0 else 40 
# #             ax.text(new_x, new_y + y_offset, text_str, color='black', fontsize=14, fontweight='bold',
# #                     ha='center', va='center', zorder=6,
# #                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor="none", alpha=0.8))

# #         fig.savefig(path_probes, pad_inches=0)
# #         plt.close(fig)

# # # ================= 3. 数据加载与处理 =================

# # def load_depth_data(file_path: Path):
# #     try:
# #         if file_path.suffix.lower() == '.npy':
# #             data = np.load(file_path)
# #             if data.ndim == 3 and data.shape[0] == 1: data = data[0]
# #             elif data.ndim == 3 and data.shape[2] == 1: data = data[:, :, 0]
# #             if data.ndim == 3 and data.shape[2] == 3: return None
# #             return data.astype(np.float32)
# #         elif file_path.suffix.lower() == '.png':
# #             data = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
# #             if data is None: return None
# #             return data.astype(np.float32) / 1000.0
# #     except Exception as e:
# #         print(f"❌ 读取异常 {file_path}: {e}")
# #     return None

# # def resolve_depth_paths(targets):
# #     resolved = []
# #     if isinstance(targets, (str, Path)): targets = [targets]
# #     for t in targets:
# #         p = Path(t).resolve()
# #         if p.is_file() and p.suffix.lower() in ['.npy', '.png']: resolved.append(p)
# #     return list(dict.fromkeys(resolved))

# # if __name__ == "__main__":
# #     print(f"{'='*60}\n📊 自适应色带截取渲染器 (量级差变红 + 完美细节)\n{'='*60}")
    
# #     target_paths = resolve_depth_paths(INPUT_TARGETS)
# #     if not target_paths: sys.exit(1)
        
# #     out_dir = Path(OUTPUT_DIR)
# #     out_dir.mkdir(parents=True, exist_ok=True)
    
# #     loaded_datasets = []
# #     img_shape = None
# #     global_mask = None

# #     for file_path in target_paths:
# #         data = load_depth_data(file_path)
# #         if data is None: continue
# #         loaded_datasets.append((file_path, data))
# #         mask = get_valid_mask(data)
# #         if img_shape is None:
# #             img_shape = data.shape
# #             global_mask = mask
# #         elif data.shape == img_shape:
# #             global_mask &= mask

# #     # 提取第一张图 (GT) 的最大值作为全局锚点标尺
# #     base_depth = loaded_datasets[0][1]
# #     base_mask = get_valid_mask(base_depth)
# #     global_vmax = np.percentile(base_depth[base_mask], 99)
# #     print(f"⚓ 全局真值上限提取完毕: {global_vmax:.2f}m")

# #     if USE_MANUAL_PROBES and MANUAL_PROBES:
# #         global_probes = []
# #         h_orig, w_orig = img_shape if img_shape else (960, 1280)
# #         for px, py in MANUAL_PROBES:
# #             px = max(0, min(px, w_orig - 1))
# #             py = max(0, min(py, h_orig - 1))
# #             pt = get_nearest_valid_point(px, py, global_mask)
# #             if pt is not None: global_probes.append(pt)
# #     else:
# #         h, w = img_shape if img_shape else (1080, 1920)
# #         cx, cy = w // 2, h // 2
# #         offset_x = w // 4
# #         pt1 = get_nearest_valid_point(cx - offset_x, cy, global_mask) if global_mask is not None else (cx - offset_x, cy)
# #         pt2 = get_nearest_valid_point(cx + offset_x, cy, global_mask) if global_mask is not None else (cx + offset_x, cy)
# #         global_probes = [pt for pt in [pt1, pt2] if pt is not None]

# #     renderer = DepthRenderer()
# #     for idx, (file_path, data) in enumerate(tqdm(loaded_datasets, desc="Rendering")):
# #         unique_stem = f"{idx:02d}_{file_path.parent.name}_{file_path.stem}"
# #         # 传入 global_vmax 给内部进行截取计算
# #         renderer.render_trio(
# #             depth_array=data,
# #             stem=unique_stem,
# #             out_dir=out_dir,
# #             target_probes=global_probes,
# #             global_vmax=global_vmax
# #         )
            
# #     print(f"\n✅ 渲染完成！输出目录: {out_dir}")