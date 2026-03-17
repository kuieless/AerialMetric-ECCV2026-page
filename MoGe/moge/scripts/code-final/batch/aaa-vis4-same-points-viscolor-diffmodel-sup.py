

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 1. 用户配置区域 =================
# ================= 1. 用户配置更新 =================
# ================= 1. 用户配置更新 (中心区域和距离约束) =================
N_BEST_PROBES = 2          # 最终显示的探针数量 (必须为2)
AUTO_PROBE_GRID = 12       # 采样网格密度 (12x12 = 144个候选点，分别在左右区域采样)
HORIZONTAL_PADDING_RATIO = 0.15 # 横向边缘留白 (图像宽度的 15%)
VERTICAL_PADDING_RATIO = 0.2  # 纵向边缘留白 (图像高度的 20%)

# 如果你想保留手动定义的点作为备选，可以保留，否则设为空
MANUAL_PROBES = []
HORIZONTAL_PADDING = 0.15  # 横向留白
VERTICAL_PADDING = 0.15    # 纵向留白
REGION_WIDTH = 0.35        # 每个采样区的宽度占比
REGION_HEIGHT = 0.35       # 每个采样区的高度占比

TARGET_SIZE = (1280, 960) 
COLORMAP = 'Spectral'
INVALID_COLOR = 'white' 
MANUAL_PROBES = [(320, 480), (880, 400)]
USE_MANUAL_PROBES = True

# 基础输出目录
BASE_OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/figsup-o2batch-wild"


# ID_LIST = [
#     # "000025", "000162", "2b782663e09916084bda251b0737f1faa4f69c51",
#     # "000037", "000163", "573ac60bfc0ac37a2323fae9b70dcec1de510efe",
#     # "000046", "0bb98b5f88653b0b274d48ebecfcc123d8b0efc7", "659a38944bba2c89dfd333e458bcf92c49741918",
#     # "000064", "0eec3b6ca59a3abe8eed525f8f57f971adfc3399", "680e2f21da1dc8529415cc8fbabdaf151d91b018",
#     # "000075", "1658132052.639874860",
#     # "000084", "1658132053.141801244", "696de3ab3098d514c2b695727f513db1f682f498",
#     # "000109", "1658132053.639567866", "738f7741a53487ba1e7ee199654bdacc716b62e5",
#     # "000121", "1658132225.135099271", "779e1f98a3c27dd05ac82f4f41239475427d6fa0",
#     # "000124", "1cbbdc58f44ef90a85e749d4c0797e71c7690610", "803a87e3def8f1b2bbf2c54c6de750dc465152ed",
#     # "000132", "221cede73c2d810f81a982a55307874439280481", "851ecf2bbb26344df9d6cd1f92d2219a43e3f9e2",
#     # "000153", "2a083087f447be8a23ee450d6a7862515e2a3870"
#     #  "689f3843bf5667ed2a430de878c15490563144ce",
# "689f3843bf5667ed2a430de878c15490563144ce",



# ]

# ID_LIST = [

#     "000293", "5e8291f9879e9f509ca107af4543d194f71399d8",
#     "000311", "5ede5f1e1184db3d2d595ee366fca9130e534eac",
#     "000321", "65fde22dc0b8282b05d0fa34c4fd8c6de96fe28a",
#     "000337", "70e684e679673794d0182eee384d4d0257e0dc8c",
#     "000383", "71e4914c99ca5e7aaf97e3707380134ebbc0a498",
#     "000388", "75ee0caba2f913b94e74f1836a53c4610385c6af",
#     "000389", "7b6be25ac5b5e573d857fb03452cc8aeea39a309",
#     "000397", "80e2010d7cf538a68f80c54b82c55438f7902978",
#     "056fc57cf43586c8791f324d5a4e0b5e28f8e62e", "80efd004099e7dc4589f2ed5c637319c1fdf08b4",
#     "077f6fa880561fa6b0340cb975669cbe1cecdb91", "81b6ddb64d781d0e7f730e9145337f63c8036197",
#     "49fe615f021044eba17e2210a54e804233eeb7ba", "91a2d941b70cf52b72ebcb6b3b4bcb94aca2baae",
#     "5a81d3d90c45e88d31fa985c103096e3bc1fd093", "96c36791d93843082d4342bd02ee67bd39776561",
#     "5e6205e82bf8d7a2416ffb2c0ce24209d9550ef2", "98c8b0a4058bd1acb8aefef54bbe42d80942c8b4"

# ]
ID_LIST = [
    "scene_041_frame_009214",
    "scene_068_frame_015753",
    "TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_s128_f012274",
    "TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_s184_f017638",
    "videoplayback5_s013_f001716",
    "videoplayback5_s014_f001924",
    "videoplayback8_s021_f003983",
    "videoplayback8_s022_f004149",
    "videoplayback8_s023_f004331",
    "videoplayback8_s035_f006583",
    "videoplayback8_s036_f006742"
]

# 定义路径模板（基于你提供的 -o 组格式）
def get_paths_for_id(image_id):
    stem = Path(image_id).stem # 移除可能存在的 .npy 或 .jpg 后缀
    # return [
    #     f"/data1/szq/Val/Viss/Sup/{stem}.npy", 
    #     f"/data1/szq/Val/Viss/Sup/mogeout-b/Baseline/{stem}/depth.npy", 
    #     f"/data1/szq/Val/Viss/Sup/mogeout-b/LoRA_Rank96/{stem}/depth.npy", 
    #     f"/data1/yzy/Benchmark/infer-photo/sup/uni/{stem}.npy", 
    #     f"/data1/yzy/Benchmark/infer-photo/sup/depth_pro_results/{stem}.npy", 
    #     f"/data1/yzy/Benchmark/infer-photo/sup/zoedepth/{stem}.npy", 
    # ]
    # return [
    #     f"/data1/szq/Val/Viss/Sup/oblique/{stem}.npy", 
    #     f"/data1/szq/Val/Viss/Sup/mogeout-o/Baseline/{stem}/depth.npy", 
    #     f"/data1/szq/Val/Viss/Sup/mogeout-o/LoRA_Rank96/{stem}/depth.npy", 
    #     f"/data1/yzy/Benchmark/infer-photo/sup/uni-o/{stem}.npy", 
    #     f"/data1/yzy/Benchmark/infer-photo/sup/depth_pro_results-o/{stem}.npy", 
    #     f"/data1/yzy/Benchmark/infer-photo/sup/zoedepth-o/{stem}.npy", 
    # ]
    return [
        f"/data1/szq/Val/Viss/Sup/wild/{stem}.npy", 
        f"/data1/szq/Val/Viss/Sup/mogeout-wild/Baseline/{stem}/depth.npy", 
        f"/data1/szq/Val/Viss/Sup/mogeout-wild/LoRA_Rank96/{stem}/depth.npy", 
        f"/data1/yzy/Benchmark/infer-photo/uni-wild/{stem}.npy", 
        f"/data1/yzy/Benchmark/infer-photo/depth_pro_results-wild/{stem}.npy", 
        # f"/data1/yzy/Benchmark/infer-photo/sup/zoedepth-o2/{stem}.npy", 
    ]

# ================= 2. 核心函数 (保持不变) =================

def get_valid_mask(depth_array):
    return np.isfinite(depth_array) & (depth_array > 1e-3) & (depth_array < 1000.0)

def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
    if valid_mask[tgt_y, tgt_x]: return (tgt_x, tgt_y)
    ys, xs = np.where(valid_mask)
    if len(ys) == 0: return None 
    dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
    best_idx = np.argmin(dist_sq)
    return (int(xs[best_idx]), int(ys[best_idx]))
# def find_best_probes(loaded_data, grid_size=12, n_best=2):
#     """
#     在 LoRA 结果与 GT 最接近的区域自动寻找探针
#     增加了左半部分和右半部分区域、以及边缘留白的约束
#     loaded_data[0] 认为是 GT, loaded_data[2] 认为是 LoRA
#     """
#     if len(loaded_data) < 3:
#         return []

#     gt_data = loaded_data[0][1]    # 第一个路径通常是 GT
#     lora_data = loaded_data[2][1]  # 对应你列表中的 LoRA_Rank96
    
#     h, w = gt_data.shape
#     valid_mask = get_valid_mask(gt_data) & get_valid_mask(lora_data)

#     # 1. 定义边缘留白
#     pad_w = int(w * HORIZONTAL_PADDING_RATIO)
#     pad_h = int(h * VERTICAL_PADDING_RATIO)

#     # 2. 定义左半部分和右半部分采样区域
#     # 左中心区域
#     left_start_x, left_end_x = pad_w, int(w / 2) - int(pad_w / 2)
#     left_start_y, left_end_y = pad_h, h - pad_h
    
#     # 右中心区域
#     right_start_x, right_end_x = int(w / 2) + int(pad_w / 2), w - pad_w
#     right_start_y, right_end_y = pad_h, h - pad_h
    
#     # 定义区域掩码
#     left_center_mask = np.zeros_like(gt_data, dtype=bool)
#     left_center_mask[left_start_y:left_end_y, left_start_x:left_end_x] = True
    
#     right_center_mask = np.zeros_like(gt_data, dtype=bool)
#     right_center_mask[right_start_y:right_end_y, right_start_x:right_end_x] = True

#     # 3. 在独立区域内进行采样和排序
    
#     def get_best_probe_in_region(region_mask, region_name):
#         final_search_mask = valid_mask & region_mask
        
#         if np.sum(final_search_mask) == 0:
#             print(f"⚠️ {region_name} 区域内未找到有效像素，尝试在区域内全图搜索")
#             final_search_mask = region_mask # 兜底，不一定有效

#         candidates = []
        
#         # 获取区域边界用于采样
#         y_indices, x_indices = np.where(region_mask)
#         if len(x_indices) == 0: return None
        
#         r_start_x, r_end_x = np.min(x_indices), np.max(x_indices)
#         r_start_y, r_end_y = np.min(y_indices), np.max(y_indices)
        
#         # 生成区域内的网格采样点
#         r_w, r_h = r_end_x - r_start_x, r_end_y - r_start_y
#         ys, xs = np.mgrid[r_start_y : r_end_y : int(r_h/(grid_size+1)), 
#                           r_start_x : r_end_x : int(r_w/(grid_size+1))]
        
#         # probe_coords = np.vstack((xs.flatten(), yy.flatten())).T
#         # 将 yy 改为 ys
#         probe_coords = np.vstack((xs.flatten(), ys.flatten())).T

#         for px, py in probe_coords:
#             px, py = int(px), int(py)
#             # 寻找最近的有效点
#             pt = get_nearest_valid_point(px, py, final_search_mask)
#             if pt:
#                 tx, ty = pt
#                 # 计算绝对误差
#                 error = abs(gt_data[ty, tx] - lora_data[ty, tx])
#                 candidates.append({
#                     'pos': (tx, ty),
#                     'error': error,
#                     'gt_val': gt_data[ty, tx],
#                     'lora_val': lora_data[ty, tx]
#                 })
        
#         # 按误差从小到大排序
#         candidates.sort(key=lambda x: x['error'])
        
#         if candidates:
#             best_c = candidates[0]
#             print(f"🔍 {region_name} 探测结果: Pos(x={best_c['pos'][0]}, y={best_c['pos'][1]}) | GT: {best_c['gt_val']:.2f}m | LoRA: {best_c['lora_val']:.2f}m | Error: {best_c['error']:.4f}m")
#             return best_c['pos']
#         else:
#             print(f"❌ {region_name} 区域内未找到合适的采样点")
#             return None

#     # 4. 获取左和右区域的最佳点
#     selected_probes = []
    
#     left_probe = get_best_probe_in_region(left_center_mask, "Left Center")
#     if left_probe: selected_probes.append(left_probe)
    
#     right_probe = get_best_probe_in_region(right_center_mask, "Right Center")
#     if right_probe: selected_probes.append(right_probe)
            
#     # 如果没选够两个点（罕见情况，例如某半边全是无效值），则忽略区域约束强制补齐
#     if len(selected_probes) < n_best:
#         print(f"⚠️ 未选够 {n_best} 个点，尝试在全图寻找备选点")
#         # 这部分逻辑可以根据需要增加，比如从 Rank 3 开始选点...
#         # 简单起见，这里假设左右两边都能选出点

#     return selected_probes
def find_best_probes(loaded_data, grid_size=12, n_best=2):
    if len(loaded_data) < 3: return []

    gt_data = loaded_data[0][1]    # 基准 GT
    lora_data = loaded_data[2][1]  # 核心对比模型 LoRA
    h, w = gt_data.shape
    valid_mask = get_valid_mask(gt_data) & get_valid_mask(lora_data)

    # 定义两个对角线区域
    # 1. 左中上区域 (Left-Mid-Top)
    l_t_coords = {
        'x1': int(w * HORIZONTAL_PADDING),
        'x2': int(w * (HORIZONTAL_PADDING + REGION_WIDTH)),
        'y1': int(h * VERTICAL_PADDING),
        'y2': int(h * (VERTICAL_PADDING + REGION_HEIGHT))
    }

    # 2. 右中下区域 (Right-Mid-Bottom)
    r_b_coords = {
        'x1': int(w * (1 - HORIZONTAL_PADDING - REGION_WIDTH)),
        'x2': int(w * (1 - HORIZONTAL_PADDING)),
        'y1': int(h * (1 - VERTICAL_PADDING - REGION_HEIGHT)),
        'y2': int(h * (1 - VERTICAL_PADDING))
    }

    def get_best_in_box(coords, name):
        # 创建局部掩码
        mask = np.zeros_like(gt_data, dtype=bool)
        mask[coords['y1']:coords['y2'], coords['x1']:coords['x2']] = True
        final_mask = valid_mask & mask
        
        if np.sum(final_mask) == 0:
            return None

        # 在矩形框内网格采样
        xs_list = np.linspace(coords['x1'], coords['x2'], grid_size).astype(int)
        ys_list = np.linspace(coords['y1'], coords['y2'], grid_size).astype(int)
        xs, ys = np.meshgrid(xs_list, ys_list)
        
        candidates = []
        for px, py in zip(xs.flatten(), ys.flatten()):
            pt = get_nearest_valid_point(px, py, final_mask)
            if pt:
                tx, ty = pt
                error = abs(gt_data[ty, tx] - lora_data[ty, tx])
                candidates.append({'pos': (tx, ty), 'error': error, 'v': gt_data[ty, tx], 'lv': lora_data[ty, tx]})
        
        candidates.sort(key=lambda x: x['error'])
        if candidates:
            c = candidates[0]
            print(f"📍 {name}: Pos({c['pos'][0]},{c['pos'][1]}) | GT: {c['v']:.2f}m | LoRA: {c['lv']:.2f}m | Err: {c['error']:.4f}")
            return c['pos']
        return None

    selected = []
    p1 = get_best_in_box(l_t_coords, "左中上 (Top-Left)")
    if p1: selected.append(p1)
    
    p2 = get_best_in_box(r_b_coords, "右中下 (Bottom-Right)")
    if p2: selected.append(p2)

    return selected
class DepthRenderer:
    def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
        self.cmap = plt.get_cmap(cmap_name).copy()
        self.cmap.set_bad(color=invalid_color) 

    def render_trio(self, depth_array, stem, out_dir, target_probes, global_vmax):
        mask = get_valid_mask(depth_array)
        if not np.any(mask): 
            print(f"⚠️ 跳过渲染 {stem}: 无有效像素")
            return
            
        depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        
        # # 1. 获取当前图的自身量程
        # valid_pixels = depth_resized[mask_resized]
        # local_vmax = np.percentile(valid_pixels, 99)
        
        # # --- 鲁棒性改进：防止除以零或极小值 ---
        # safe_global_vmax = max(global_vmax, 1e-3)
        # 1. 获取当前图的自身量程
        valid_pixels = depth_resized[mask_resized]
        
        # 🌟 修复核心：强制过滤掉 cv2.resize 插值产生的 NaN 和 Inf
        valid_pixels = valid_pixels[np.isfinite(valid_pixels)]
        
        if len(valid_pixels) == 0:
            print(f"⚠️ {stem}: 缩放后无有效像素，跳过渲染")
            return
            
        local_vmax = np.percentile(valid_pixels, 99)
        
        # --- 鲁棒性改进：防止除以零或极小值 ---
        safe_global_vmax = max(global_vmax, 1e-3)
        safe_local_vmax = max(local_vmax, 1e-3)
        
        # 2. 计算动态带宽
        ratio = min(safe_local_vmax / safe_global_vmax, 1.0)
        band = 0.3 + 0.7 * ratio 
        
        # 3. 反推 plot_vmax，并增加硬性保护
        plot_vmax = safe_local_vmax / band
        if not np.isfinite(plot_vmax) or plot_vmax <= 0:
            plot_vmax = safe_local_vmax # 退化处理
        
        # 4. 准备渲染数据
        depth_clipped = np.clip(depth_resized, 0, plot_vmax)
        # 将无效值设为 NaN 以便显示 invalid_color
        depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        
        # --- 调试信息：如果遇到全白，观察控制台输出 ---
        if np.isnan(depth_masked).all():
            print(f"❌ 警告: {stem} 渲染数据全为 NaN")

        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')

        # 核心渲染
        ax.imshow(depth_masked, cmap=self.cmap, vmin=0, vmax=plot_vmax, 
                  extent=[0, TARGET_SIZE[0], TARGET_SIZE[1], 0])

        fig.savefig(out_dir / f"{stem}_pure.png", pad_inches=0)

        # --- 画探针 (这部分保持不变) ---
        orig_h, orig_w = depth_array.shape
        scale_x, scale_y = TARGET_SIZE[0] / orig_w, TARGET_SIZE[1] / orig_h
        colors = ['#FF3366', '#00E676'] 
        for pt_idx, (px, py) in enumerate(target_probes):
            new_x, new_y = int(px * scale_x), int(py * scale_y)
            c = colors[pt_idx % len(colors)]
            ax.plot(new_x, new_y, marker='o', markerfacecolor=c, markeredgecolor='white', 
                    markersize=12, markeredgewidth=2, zorder=5)
            
            # 探针取值也要防错
            val = depth_resized[new_y, new_x]
            val_str = f"{val:.1f}m" if np.isfinite(val) else "N/A"
            
            ax.text(new_x, new_y + (-40 if pt_idx == 0 else 40), val_str, 
                    color='black', fontsize=14, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        fig.savefig(out_dir / f"{stem}_probes.png", pad_inches=0)
        plt.close(fig)

def load_depth_data(file_path):
    p = Path(file_path)
    if not p.exists(): return None
    try:
        data = np.load(p)
        if data.ndim == 3: data = data.squeeze()
        return data.astype(np.float32)
    except: return None

# ================= 3. 批量执行逻辑 =================

# if __name__ == "__main__":
#     renderer = DepthRenderer()
    
#     # 对 ID 列表进行去重并遍历
#     unique_ids = list(dict.fromkeys([Path(i).stem for i in ID_LIST]))
    
#     print(f"🚀 开始批量处理，共 {len(unique_ids)} 组数据...")

#     for current_id in tqdm(unique_ids, desc="Total Progress"):
#         # 1. 为这一组创建独立的文件夹
#         group_out_dir = Path(BASE_OUTPUT_DIR) / current_id
#         group_out_dir.mkdir(parents=True, exist_ok=True)
        
#         # 2. 获取这组的 6 个路径
#         target_paths = get_paths_for_id(current_id)
        
#         # 3. 加载数据并计算全局标尺
#         loaded_data = []
#         global_mask = None
        
#         for p in target_paths:
#             d = load_depth_data(p)
#             if d is not None:
#                 m = get_valid_mask(d)
#                 loaded_data.append((Path(p), d, m))
#                 global_mask = m if global_mask is None else global_mask & m
        
#         if not loaded_data:
#             print(f"⚠️ 跳过 ID {current_id}: 未找到有效数据")
#             continue

#         # 以第一张图（真值）计算 global_vmax
#         base_depth, base_mask = loaded_data[0][1], loaded_data[0][2]
#         global_vmax = np.percentile(base_depth[base_mask], 99)

#         # 4. 计算探针位置
#         h, w = loaded_data[0][1].shape
#         global_probes = []
#         for px, py in MANUAL_PROBES:
#             pt = get_nearest_valid_point(px, py, global_mask)
#             if pt: global_probes.append(pt)

#         # 5. 渲染这组的 6 张图
#         for idx, (f_path, data, _) in enumerate(loaded_data):
#             # 命名格式：序号_方法名
#             # f_path.parent.name 会获取类似 "Baseline" 或 "uni-o" 的文件夹名
#             method_name = f_path.parent.name if "mogeout" not in str(f_path) else f"{f_path.parent.parent.name}_{f_path.parent.name}"
#             unique_stem = f"{idx:02d}_{method_name}"
            
#             renderer.render_trio(
#                 depth_array=data,
#                 stem=unique_stem,
#                 out_dir=group_out_dir,
#                 target_probes=global_probes,
#                 global_vmax=global_vmax
#             )
# if __name__ == "__main__":
#     renderer = DepthRenderer()
    
#     unique_ids = list(dict.fromkeys([Path(i).stem for i in ID_LIST]))
#     print(f"🚀 开始批量处理，共 {len(unique_ids)} 组数据...")

#     for current_id in tqdm(unique_ids, desc="Total Progress"):
#         group_out_dir = Path(BASE_OUTPUT_DIR) / current_id
#         group_out_dir.mkdir(parents=True, exist_ok=True)
        
#         target_paths = get_paths_for_id(current_id)
        
#         loaded_data = []
#         base_shape = None  # 用于对齐的基准尺寸
#         global_mask = None
        
#         for p in target_paths:
#             d = load_depth_data(p)
#             if d is not None:
#                 # --- 新增逻辑：处理尺寸不一致 ---
#                 if base_shape is None:
#                     base_shape = d.shape  # 以第一张图(通常是GT)为基准
#                 elif d.shape != base_shape:
#                     # 如果尺寸不符，强制 resize 到基准尺寸 (使用最近邻插值保持深度值特征)
#                     d = cv2.resize(d, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
                
#                 m = get_valid_mask(d)
#                 loaded_data.append((Path(p), d, m))
                
#                 if global_mask is None:
#                     global_mask = m
#                 else:
#                     global_mask = global_mask & m
        
#         if not loaded_data:
#             print(f"⚠️ 跳过 ID {current_id}: 未找到有效数据")
#             continue

#         # 后面渲染逻辑保持不变...
#         base_depth, base_mask = loaded_data[0][1], loaded_data[0][2]
#         global_vmax = np.percentile(base_depth[base_mask], 99)

#         # 提取探针位置
#         h_orig, w_orig = base_shape
#         global_probes = []
#         for px, py in MANUAL_PROBES:
#             # 确保探针坐标不越界
#             safe_x = max(0, min(int(px), w_orig - 1))
#             safe_y = max(0, min(int(py), h_orig - 1))
#             pt = get_nearest_valid_point(safe_x, safe_y, global_mask)
#             if pt: global_probes.append(pt)

#         for idx, (f_path, data, _) in enumerate(loaded_data):
#             # 获取方法名
#             parent_name = f_path.parent.name
#             if "mogeout" in str(f_path):
#                 method_name = f"{f_path.parent.parent.name}_{f_path.parent.name}"
#             else:
#                 method_name = parent_name
            
#             unique_stem = f"{idx:02d}_{method_name}"
            
#             renderer.render_trio(
#                 depth_array=data,
#                 stem=unique_stem,
#                 out_dir=group_out_dir,
#                 target_probes=global_probes,
#                 global_vmax=global_vmax
#             )

#     print(f"\n✅ 所有任务完成！根目录: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    renderer = DepthRenderer()
    unique_ids = list(dict.fromkeys([Path(i).stem for i in ID_LIST]))
    
    for current_id in tqdm(unique_ids, desc="Total Progress"):
        group_out_dir = Path(BASE_OUTPUT_DIR) / current_id
        group_out_dir.mkdir(parents=True, exist_ok=True)
        
        target_paths = get_paths_for_id(current_id)
        loaded_data = []
        base_shape = None
        
        # 加载数据
        for p in target_paths:
            d = load_depth_data(p)
            if d is not None:
                if base_shape is None: base_shape = d.shape
                elif d.shape != base_shape:
                    d = cv2.resize(d, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
                loaded_data.append((Path(p), d, get_valid_mask(d)))

        if len(loaded_data) < 3:
            print(f"⚠️ {current_id} 数据不足，无法进行误差对比采样")
            continue

        # --- 核心逻辑：自动寻找最接近的探针 ---
        global_probes = find_best_probes(loaded_data, grid_size=AUTO_PROBE_GRID, n_best=N_BEST_PROBES)

        # 后面渲染逻辑保持不变
        base_depth, base_mask = loaded_data[0][1], loaded_data[0][2]
        
        # 🌟 修复全局标尺：同样清洗有效像素
        base_valid_pixels = base_depth[base_mask]
        base_valid_pixels = base_valid_pixels[np.isfinite(base_valid_pixels)]
        
        if len(base_valid_pixels) > 0:
            global_vmax = np.percentile(base_valid_pixels, 99)
        else:
            global_vmax = 100.0 # 极端情况的兜底值

        for idx, (f_path, data, _) in enumerate(loaded_data):
            parent_name = f_path.parent.name
            method_name = f"{f_path.parent.parent.name}_{f_path.parent.name}" if "mogeout" in str(f_path) else parent_name
            unique_stem = f"{idx:02d}_{method_name}"
            
            renderer.render_trio(
                depth_array=data,
                stem=unique_stem,
                out_dir=group_out_dir,
                target_probes=global_probes,
                global_vmax=global_vmax
            )

    print(f"\n✅ 处理完成！探针已对齐 GT 与 LoRA 最接近的区域。")