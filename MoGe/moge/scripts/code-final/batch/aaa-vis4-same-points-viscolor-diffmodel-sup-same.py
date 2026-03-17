# import os
# import sys
# import numpy as np
# import cv2
# from pathlib import Path
# from tqdm import tqdm
# import matplotlib
# matplotlib.use('Agg')
# # import matplotlib.subplots as plt_subplots
# import matplotlib.pyplot as plt

# # ================= 1. 用户配置区域 =================
# TARGET_SIZE = (1280, 960) # (宽, 高)
# INPUT_TARGETS = [
# # "/data1/szq/Val/Viss/Sup/2f6ff850a107cda8e3b9968f8c86c1d5a280c452.npy", 
# # "/data1/szq/Val/Viss/Sup/mogeout-b/Baseline/2f6ff850a107cda8e3b9968f8c86c1d5a280c452/depth.npy", 
# # "/data1/szq/Val/Viss/Sup/mogeout-b/LoRA_Rank96/2f6ff850a107cda8e3b9968f8c86c1d5a280c452/depth.npy", 
# # "/data1/yzy/Benchmark/infer-photo/sup/uni/2f6ff850a107cda8e3b9968f8c86c1d5a280c452.npy", 
# # "/data1/yzy/Benchmark/infer-photo/sup/depth_pro_results/2f6ff850a107cda8e3b9968f8c86c1d5a280c452.npy", 
# # "/data1/yzy/Benchmark/infer-photo/sup/zoedepth/2f6ff850a107cda8e3b9968f8c86c1d5a280c452.npy", 

# "/data1/szq/Val/Viss/Sup/oblique/0bb98b5f88653b0b274d48ebecfcc123d8b0efc7.npy", 
# "/data1/szq/Val/Viss/Sup/mogeout-o/Baseline/0bb98b5f88653b0b274d48ebecfcc123d8b0efc7/depth.npy", 
# "/data1/szq/Val/Viss/Sup/mogeout-o/LoRA_Rank96/0bb98b5f88653b0b274d48ebecfcc123d8b0efc7/depth.npy", 
# "/data1/yzy/Benchmark/infer-photo/sup/uni-o/0bb98b5f88653b0b274d48ebecfcc123d8b0efc7.npy", 
# "/data1/yzy/Benchmark/infer-photo/sup/depth_pro_results-o/0bb98b5f88653b0b274d48ebecfcc123d8b0efc7.npy", 
# "/data1/yzy/Benchmark/infer-photo/sup/zoedepth-o/0bb98b5f88653b0b274d48ebecfcc123d8b0efc7.npy", 


# ]

# OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/figsup-o1" 
# MANUAL_PROBES = [
#     (320, 480),
#     (880, 400) 
# ]
# USE_MANUAL_PROBES = True
# COLORMAP = 'Spectral'       
# INVALID_COLOR = 'white'     

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

#     def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, target_probes: list, global_vmax: float):
#         mask = get_valid_mask(depth_array)
#         if not np.any(mask): return
            
#         depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
#         mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        
#         # 1. 获取当前图的自身量程 (99% 避免噪点)
#         valid_pixels = depth_resized[mask_resized]
#         local_vmax = np.percentile(valid_pixels, 99)
        
#         # 2. 核心魔法：计算动态带宽
#         ratio = min(local_vmax / global_vmax, 1.0)
#         # 强制保底 30% 的色带宽度，确保有细节 (可按喜好微调 0.3)
#         band = 0.3 + 0.7 * ratio 
        
#         # 3. 反推 matplotlib 需要的 vmax
#         plot_vmax = local_vmax / band
        
#         depth_clipped = np.clip(depth_resized, 0, plot_vmax)
#         depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        
#         path_pure = out_dir / f"{stem}_pure.png"
#         path_probes = out_dir / f"{stem}_probes.png"

#         # 使用防抖画布
#         fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
#         fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
#         ax.set_xlim(0, TARGET_SIZE[0])
#         ax.set_ylim(TARGET_SIZE[1], 0)
#         ax.axis('off')

#         # 统一以 0 为起点，plot_vmax 为终点
#         ax.imshow(depth_masked, cmap=self.cmap, vmin=0, vmax=plot_vmax, 
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

# # ================= 3. 数据加载与处理 =================

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

# def resolve_depth_paths(targets):
#     resolved = []
#     if isinstance(targets, (str, Path)): targets = [targets]
#     for t in targets:
#         p = Path(t).resolve()
#         if p.is_file() and p.suffix.lower() in ['.npy', '.png']: resolved.append(p)
#     return list(dict.fromkeys(resolved))

# if __name__ == "__main__":
#     print(f"{'='*60}\n📊 自适应色带截取渲染器 (量级差变红 + 完美细节)\n{'='*60}")
    
#     target_paths = resolve_depth_paths(INPUT_TARGETS)
#     if not target_paths: sys.exit(1)
        
#     out_dir = Path(OUTPUT_DIR)
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     loaded_datasets = []
#     img_shape = None
#     global_mask = None

#     for file_path in target_paths:
#         data = load_depth_data(file_path)
#         if data is None: continue
#         loaded_datasets.append((file_path, data))
#         mask = get_valid_mask(data)
#         if img_shape is None:
#             img_shape = data.shape
#             global_mask = mask
#         elif data.shape == img_shape:
#             global_mask &= mask

#     # 提取第一张图 (GT) 的最大值作为全局锚点标尺
#     base_depth = loaded_datasets[0][1]
#     base_mask = get_valid_mask(base_depth)
#     global_vmax = np.percentile(base_depth[base_mask], 99)
#     print(f"⚓ 全局真值上限提取完毕: {global_vmax:.2f}m")

#     if USE_MANUAL_PROBES and MANUAL_PROBES:
#         global_probes = []
#         h_orig, w_orig = img_shape if img_shape else (960, 1280)
#         for px, py in MANUAL_PROBES:
#             px = max(0, min(px, w_orig - 1))
#             py = max(0, min(py, h_orig - 1))
#             pt = get_nearest_valid_point(px, py, global_mask)
#             if pt is not None: global_probes.append(pt)
#     else:
#         h, w = img_shape if img_shape else (1080, 1920)
#         cx, cy = w // 2, h // 2
#         offset_x = w // 4
#         pt1 = get_nearest_valid_point(cx - offset_x, cy, global_mask) if global_mask is not None else (cx - offset_x, cy)
#         pt2 = get_nearest_valid_point(cx + offset_x, cy, global_mask) if global_mask is not None else (cx + offset_x, cy)
#         global_probes = [pt for pt in [pt1, pt2] if pt is not None]

#     renderer = DepthRenderer()
#     for idx, (file_path, data) in enumerate(tqdm(loaded_datasets, desc="Rendering")):
#         unique_stem = f"{idx:02d}_{file_path.parent.name}_{file_path.stem}"
#         # 传入 global_vmax 给内部进行截取计算
#         renderer.render_trio(
#             depth_array=data,
#             stem=unique_stem,
#             out_dir=out_dir,
#             target_probes=global_probes,
#             global_vmax=global_vmax
#         )
            
#     print(f"\n✅ 渲染完成！输出目录: {out_dir}")

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
N_BEST_PROBES = 2         # 最终显示的探针数量 (必须为2)
AUTO_PROBE_GRID = 12       # 采样网格密度 (12x12 = 144个候选点，分别在左右区域采样)
HORIZONTAL_PADDING_RATIO = 0.10 # 横向边缘留白 (图像宽度的 15%)
VERTICAL_PADDING_RATIO = 0.1  # 纵向边缘留白 (图像高度的 20%)

# 如果你想保留手动定义的点作为备选，可以保留，否则设为空
MANUAL_PROBES = []
HORIZONTAL_PADDING = 0.1  # 横向留白
VERTICAL_PADDING = 0.1    # 纵向留白
REGION_WIDTH = 0.5        # 每个采样区的宽度占比
REGION_HEIGHT = 0.4       # 每个采样区的高度占比

TARGET_SIZE = (1280, 960) 
COLORMAP = 'Spectral'
INVALID_COLOR = 'white' 
MANUAL_PROBES = [(320, 480), (880, 400)]
USE_MANUAL_PROBES = True

# 基础输出目录
BASE_OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/figsup-o2batch7"


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
# ID_LIST = [
#     "12feed6c879d71cbfe615af29394095c8375ef0e", "5d8316b05cf61fd102584274ef14c4dbb88d5fef",
#     "6421cede38b0a8abae54ee3037bd52746707dcd1", "74344880b295d15334852916152850aa9a5269f5",
#     "92c170e95a5e3948e912437a3bfe03e7a6cb6a7b", "95eb4263d0d3e81727cade775bc11804b0ab0c12",
#     "faac9d8272bb932d753a5c2f37ae3c047b30c7a1"
# ]
ID_LIST = [
    "17b762848c68ee5e43a7fea9bbebba34b8b91631", "2ac1b763169f266126717c1ef7ea5014f673bc85",
    "8354b4b7a69b1809d28ed1b98e952433e1d12be6", "869ac26d8756d023cbdce5b845e89137dea6a5a0",
    "a446c6e1f5e432a3a4d23c5b9bdce71101bad7c8", "e532a7d6071fc1fa4dba3d2f390af4610dcda3d2",
    "e6dd3a0c3b9191c7110bdef5ec4a58046b19fffb", "f9f7d8526e0efcdf9a78fb55c882f06d9c69e867"
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
        f"/data1/szq/Val/Viss/Sup/oblique5/{stem}.npy", 
        f"/data1/szq/Val/Viss/Sup/mogeout-o5/Baseline/{stem}/depth.npy", 
        f"/data1/szq/Val/Viss/Sup/mogeout-o5/LoRA_Rank96/{stem}/depth.npy", 
        f"/data1/yzy/Benchmark/infer-photo/uni-o5/{stem}.npy", 
        f"/data1/yzy/Benchmark/infer-photo/depth_pro_results-o5/{stem}.npy", 
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
def find_joint_best_probes(all_groups_data, grid_size=20):
    """
    支持不同分辨率图片的全局共享探针计算
    """
    print(f"\n协同分析中... 正在为 {len(all_groups_data)} 组多尺寸数据寻找最优共享探针位置")
    
    # 1. 定义虚拟标准空间 (使用配置中的 TARGET_SIZE)
    v_w, v_h = TARGET_SIZE
    joint_consensus_mask = np.ones((v_h, v_w), dtype=bool)
    
    # 预处理：将所有组的掩码统一到虚拟空间
    standardized_groups = []
    for group in all_groups_data:
        gt_data = group[0][1]
        lora_data = group[2][1]
        
        # 计算该组的有效掩码并缩放到标准空间
        m_gt = get_valid_mask(gt_data).astype(np.uint8)
        m_lora = get_valid_mask(lora_data).astype(np.uint8)
        m_combined = cv2.resize(m_gt & m_lora, (v_w, v_h), interpolation=cv2.INTER_NEAREST) > 0
        joint_consensus_mask &= m_combined
        
        # 同时为了计算误差，把深度图也 resize 到标准空间
        gt_res = cv2.resize(gt_data, (v_w, v_h), interpolation=cv2.INTER_LINEAR)
        lora_res = cv2.resize(lora_data, (v_w, v_h), interpolation=cv2.INTER_LINEAR)
        standardized_groups.append((gt_res, lora_res))

    # 2. 定义采样区域 (在标准空间中)
    pad_w, pad_h = int(v_w * 0.15), int(v_h * 0.15)
    reg_w, reg_h = int(v_w * 0.35), int(v_h * 0.35)

    regions = {
        "Top-Left (左中上)": {
            'x1': pad_w, 'x2': pad_w + reg_w,
            'y1': pad_h, 'y2': pad_h + reg_h
        },
        "Bottom-Right (右中下)": {
            'x1': v_w - pad_w - reg_w, 'x2': v_w - pad_w,
            'y1': v_h - pad_h - reg_h, 'y2': v_h - pad_h
        }
    }

    shared_probes_normalized = [] # 存储标准空间下的 (x, y)

    for name, coords in regions.items():
        best_pt = None
        min_avg_error = float('inf')

        # 生成标准空间内的采样网格
        xs_list = np.linspace(coords['x1'], coords['x2'], grid_size).astype(int)
        ys_list = np.linspace(coords['y1'], coords['y2'], grid_size).astype(int)
        
        for py in ys_list:
            for px in xs_list:
                # 检查在所有图中是否都有效
                if not joint_consensus_mask[py, px]:
                    continue
                
                # 计算 7 组数据的平均误差
                errors = []
                for gt_res, lora_res in standardized_groups:
                    err = abs(gt_res[py, px] - lora_res[py, px])
                    errors.append(err)
                
                avg_err = np.mean(errors)
                if avg_err < min_avg_error:
                    min_avg_error = avg_err
                    best_pt = (px, py)

        if best_pt:
            print(f"✅ {name} 共享点确定 (标准空间): {best_pt} | 平均误差: {min_avg_error:.4f}m")
            shared_probes_normalized.append(best_pt)

    return shared_probes_normalized
# class DepthRenderer:
#     def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
#         self.cmap = plt.get_cmap(cmap_name).copy()
#         self.cmap.set_bad(color=invalid_color)

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
        
        # ================== 核心细节魔法 (单图独立标尺) ==================
        # 1. 获取当前图的自身量程 (99% 避免噪点)
        valid_pixels = depth_resized[mask_resized]
        if len(valid_pixels) == 0: return

        # 核心：使用该图自身的 1% 和 99% 分位数作为色带的起止点
        # 这确保了 98% 的深度变化都能获得最大的颜色对比度
        vmin = np.percentile(valid_pixels, 1)
        vmax = np.percentile(valid_pixels, 99)
        
        # 鲁棒性保护：防止 vmax <= vmin 导致渲染失败
        if vmax <= vmin:
            vmax = vmin + 1.0 
        
        # 2. 准备渲染数据 (使用独立的 vmin/vmax 进行 clip)
        depth_clipped = np.clip(depth_resized, vmin, vmax)
        # 将无效值设为 NaN 以便显示 invalid_color
        depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        # ===============================================================
        
        if np.isnan(depth_masked).all():
            print(f"❌ 警告: {stem} 渲染数据全为 NaN")

        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')

        # 核心渲染：使用单图独立的 vmin 和 vmax
        ax.imshow(depth_masked, cmap=self.cmap, vmin=vmin, vmax=vmax, 
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
            
            val = depth_resized[new_y, new_x]
            val_str = f"{val:.1f}m" if np.isfinite(val) else "N/A"
            
            # 探针文本增加 bbox 增强清晰度
            ax.text(new_x, new_y + (-40 if pt_idx == 0 else 40), val_str, 
                    color='black', fontsize=14, fontweight='bold', ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor="none", alpha=0.8))

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

# if __name__ == "__main__":
#     renderer = DepthRenderer()
#     unique_ids = list(dict.fromkeys([Path(i).stem for i in ID_LIST]))
    
#     for current_id in tqdm(unique_ids, desc="Total Progress"):
#         group_out_dir = Path(BASE_OUTPUT_DIR) / current_id
#         group_out_dir.mkdir(parents=True, exist_ok=True)
        
#         target_paths = get_paths_for_id(current_id)
#         loaded_data = []
#         base_shape = None
        
#         # 加载数据
#         for p in target_paths:
#             d = load_depth_data(p)
#             if d is not None:
#                 if base_shape is None: base_shape = d.shape
#                 elif d.shape != base_shape:
#                     d = cv2.resize(d, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
#                 loaded_data.append((Path(p), d, get_valid_mask(d)))

#         if len(loaded_data) < 3:
#             print(f"⚠️ {current_id} 数据不足，无法进行误差对比采样")
#             continue

#         # --- 核心逻辑：自动寻找最接近的探针 ---
#         global_probes = find_best_probes(loaded_data, grid_size=AUTO_PROBE_GRID, n_best=N_BEST_PROBES)

#         # 后面渲染逻辑保持不变
#         base_depth, base_mask = loaded_data[0][1], loaded_data[0][2]
#         global_vmax = np.percentile(base_depth[base_mask], 99)

#         for idx, (f_path, data, _) in enumerate(loaded_data):
#             parent_name = f_path.parent.name
#             method_name = f"{f_path.parent.parent.name}_{f_path.parent.name}" if "mogeout" in str(f_path) else parent_name
#             unique_stem = f"{idx:02d}_{method_name}"
            
#             renderer.render_trio(
#                 depth_array=data,
#                 stem=unique_stem,
#                 out_dir=group_out_dir,
#                 target_probes=global_probes,
#                 global_vmax=global_vmax
#             )

#     print(f"\n✅ 处理完成！探针已对齐 GT 与 LoRA 最接近的区域。")

if __name__ == "__main__":
    renderer = DepthRenderer()
    unique_ids = list(dict.fromkeys([Path(i).stem for i in ID_LIST]))
    
    # --- 第一步：预加载所有数据以计算共享探针 ---
    all_groups_data = []
    print("📦 正在预加载所有组数据以计算全局最优探针...")
    for current_id in unique_ids:
        target_paths = get_paths_for_id(current_id)
        loaded_group = []
        base_shape = None
        for p in target_paths:
            d = load_depth_data(p)
            if d is not None:
                if base_shape is None: base_shape = d.shape
                elif d.shape != base_shape:
                    d = cv2.resize(d, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_NEAREST)
                loaded_group.append((Path(p), d, get_valid_mask(d)))
        if len(loaded_group) >= 3:
            all_groups_data.append(loaded_group)

    # --- 第二步：寻找 7 组共享的最优探针 ---
    global_shared_probes = find_joint_best_probes(all_groups_data, grid_size=20)

    # --- 第三步：正式渲染 ---
    for i, current_id in enumerate(unique_ids):
        group_out_dir = Path(BASE_OUTPUT_DIR) / current_id
        group_out_dir.mkdir(parents=True, exist_ok=True)
        
        loaded_data = all_groups_data[i]
        
        # 使用第一张图计算该组的 global_vmax (每组的量程可以独立，但探针位置统一)
# 计算当前组的 global_vmax
        base_depth, base_mask = loaded_data[0][1], loaded_data[0][2]
        current_vmax = np.percentile(base_depth[base_mask], 99)
        
        # --- 关键：将标准空间的探针还原到当前图的原始尺寸 ---
        h_orig, w_orig = base_depth.shape
        v_w, v_h = TARGET_SIZE
        current_group_probes = []
        for (nx, ny) in global_shared_probes:
            # 按比例还原坐标
            rx = int(nx * w_orig / v_w)
            ry = int(ny * h_orig / v_h)
            current_group_probes.append((rx, ry))

        for idx, (f_path, data, _) in enumerate(loaded_data):
            parent_name = f_path.parent.name
            method_name = f"{f_path.parent.parent.name}_{f_path.parent.name}" if "mogeout" in str(f_path) else parent_name
            unique_stem = f"{idx:02d}_{method_name}"
            
            renderer.render_trio(
                depth_array=data,
                stem=unique_stem,
                out_dir=group_out_dir,
                target_probes=current_group_probes,
                global_vmax=current_vmax
            )

    print(f"\n✨ 恭喜！7 组图片已全部渲染完成，且探针位置完全一致。")