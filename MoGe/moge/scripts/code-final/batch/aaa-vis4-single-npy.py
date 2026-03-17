import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 防止在无界面服务器报错
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ================= 1. 用户配置区域 =================

# 🔥 输入路径：支持单个 .npy、文件夹，或路径列表
INPUT_TARGETS = [
    "/data1/szq/Val/vis",
] 

OUTPUT_DIR = "/data1/szq/Val/vis/wild5" # 渲染后图片的保存目录

# 可视化参数
COLORMAP = 'Spectral'       # 推荐：Spectral (近红远蓝) 或 turbo
GLOBAL_VMIN = None          # 全局最小深度 (m)。设为 None 则按单张图自动计算
GLOBAL_VMAX = None          # 全局最大深度 (m)。设为 None 则按单张图自动计算
INVALID_COLOR = 'black'     # 无效深度(NaN)或背景的颜色 (black, white, 或 transparent)

# ================= 2. 核心渲染器 =================

class DepthRenderer:
    def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
        self.cmap = plt.get_cmap(cmap_name).copy()
        self.cmap.set_bad(color=invalid_color) # 设置无效区域的颜色

    def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, vmin=None, vmax=None):
        """一次性输出三种风格的深度图"""
        
        # 1. 掩码与清理
        mask = (depth_array > 1e-3) & np.isfinite(depth_array)
        depth_masked = np.where(mask, depth_array, np.nan)
        
        if np.isnan(depth_masked).all():
            print(f"⚠️ {stem} 全是无效深度值，跳过。")
            return
            
        # 2. 计算色阶
        if vmin is None: vmin = np.nanquantile(depth_masked, 0.01)
        if vmax is None: vmax = np.nanquantile(depth_masked, 0.99)

        # 准备输出路径
        path_pure = out_dir / f"{stem}_1_pure.png"
        path_cbar = out_dir / f"{stem}_2_cbar.png"
        path_probes = out_dir / f"{stem}_3_probes.png"

        h, w = depth_array.shape
        aspect = w / h
        fig_w = max(6, 5 * aspect)

        # ==========================================
        # 🎨 输出 1: 纯粹的深度图 (1:1 原生像素导出)
        # ==========================================
        # plt.imsave 可以直接导出没有任何白边和边框的纯净图片
        plt.imsave(str(path_pure), depth_masked, cmap=self.cmap, vmin=vmin, vmax=vmax)

        # ==========================================
        # 🎨 输出 2: 深度图 + Colorbar
        # ==========================================
        # ==========================================
        # 🎨 输出 3: 深度图 + 探针探测值 (智能中心吸附版)
        # ==========================================
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        ax.imshow(depth_masked, cmap=self.cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')

        # --- 🤖 智能寻找有效探针点 ---
        def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
            """寻找距离目标坐标最近的有效深度点"""
            # 如果目标点本身有效，直接返回
            if valid_mask[tgt_y, tgt_x]:
                return (tgt_x, tgt_y)
            
            # 如果目标点无效（例如在天上），计算所有有效点到它的距离，找最近的
            ys, xs = np.where(valid_mask)
            if len(ys) == 0:
                return None # 全图无有效深度
                
            dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
            best_idx = np.argmin(dist_sq)
            return (xs[best_idx], ys[best_idx])

        # 🔧 1. 设定你理想的点位置 (目前设定为中心偏左、偏右)
        cx, cy = w // 2, h // 2       # 画面正中心
        offset_x = w // 4             # 点距离中心的远近 (w//6 比较靠近中心，w//4 会分开一点)

        pt1 = get_nearest_valid_point(cx - offset_x, cy, mask)
        pt2 = get_nearest_valid_point(cx + offset_x, cy, mask)
        
        test_points = [pt for pt in [pt1, pt2] if pt is not None]

        # --- 🎨 绘制点和悬浮文字 ---
        colors = ['#FF3366', '#00E676'] # 亮粉和亮绿
        
        for pt_idx, (x, y) in enumerate(test_points):
            c = colors[pt_idx % len(colors)]
            
            # 1. 画靶心点
            ax.plot(x, y, marker='o', markerfacecolor=c, markeredgecolor='white', 
                    markersize=14, markeredgewidth=2.5, zorder=5)
            
            val = depth_masked[y, x]
            text_str = f"{val:.1f}m"
            
            # 2. 画悬浮文字框 (取消了引线)
            # y_offset 控制气泡在点的正上方还是正下方
            y_offset = -60 if pt_idx == 0 else 40 
            
            ax.text(
                x, y + y_offset, 
                text_str,
                color='black', fontsize=15, fontweight='bold',
                ha='center', va='center', zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor="none", alpha=0.85)
            )

        plt.tight_layout()
        fig.savefig(path_probes, bbox_inches='tight', dpi=200)
        plt.close(fig)

# ================= 3. 文件解析与批处理 =================

def resolve_npy_paths(targets):
    resolved = []
    if isinstance(targets, (str, Path)):
        targets = [targets]
    for t in targets:
        p = Path(t).resolve()
        if p.is_file() and p.suffix == '.npy':
            resolved.append(p)
        elif p.is_dir():
            resolved.extend(sorted(p.rglob('*.npy')))
    return list(dict.fromkeys(resolved))

if __name__ == "__main__":
    print(f"{'='*60}\n📊 深度图三重渲染器 (Pure / Colorbar / Probes)\n{'='*60}")
    
    target_paths = resolve_npy_paths(INPUT_TARGETS)
    if not target_paths:
        print("❌ 未找到任何 .npy 文件。")
        sys.exit(1)
        
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = DepthRenderer()
    
    for npy_path in tqdm(target_paths, desc="Rendering Trios"):
        try:
            data = np.load(npy_path)
            
            if data.ndim == 3 and data.shape[0] == 1: data = data[0]
            elif data.ndim == 3 and data.shape[2] == 1: data = data[:, :, 0]
                
            # 排除法向图 (形状为 H,W,3)
            if data.ndim == 3 and data.shape[2] == 3:
                continue
                
            # 保留父目录名称前缀以防文件重名
            parent_name = npy_path.parent.name
            if parent_name == npy_path.parent.parent.name or "out" in parent_name.lower():
                stem = npy_path.stem
            else:
                stem = f"{parent_name}_{npy_path.stem}"
                
            renderer.render_trio(
                depth_array=data,
                stem=stem,
                out_dir=out_dir,
                vmin=GLOBAL_VMIN,
                vmax=GLOBAL_VMAX
            )
            
        except Exception as e:
            print(f"\n❌ 处理失败 {npy_path.name}: {e}")

    print(f"\n✅ 渲染完成！所有图片已保存至: {out_dir}")






# import os
# import sys
# import numpy as np
# import cv2
# from pathlib import Path
# from tqdm import tqdm
# import matplotlib
# matplotlib.use('Agg')  # 防止在无界面服务器报错
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# # ================= 1. 用户配置区域 =================

# # 🔥 输入路径：支持单个 .npy、文件夹，或路径列表
# INPUT_TARGETS = [
#     "/data1/szq/Val/vis",
# ] 

# OUTPUT_DIR = "/data1/szq/Val/vis/wild5" # 渲染后图片的保存目录

# # 可视化参数
# COLORMAP = 'Spectral'       # 推荐：Spectral (近红远蓝) 或 turbo
# GLOBAL_VMIN = None          # 全局最小深度 (m)。设为 None 则按单张图自动计算
# GLOBAL_VMAX = None          # 全局最大深度 (m)。设为 None 则按单张图自动计算

# # 🌟 需求 1: 无效值背景色改为纯白
# INVALID_COLOR = 'white'     # 无效深度(NaN)或背景的颜色 

# # 🌟 需求 2: 稀疏深度增强 (让稀疏点变大/变明显)
# DILATE_SPARSE = True        # 是否对稀疏深度图进行“膨胀”处理使其清晰可见
# DILATE_KERNEL = 5           # 膨胀核大小 (数字越大，稀疏点被画得越大。推荐 3, 5, 或 7)

# # ================= 2. 核心渲染器 =================

# class DepthRenderer:
#     def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
#         self.cmap = plt.get_cmap(cmap_name).copy()
#         self.cmap.set_bad(color=invalid_color) # 设置无效区域的颜色 (白色)

#     def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, vmin=None, vmax=None):
#         """一次性输出三种风格的深度图"""
        
#         # 1. 掩码与清理
#         mask = (depth_array > 1e-3) & np.isfinite(depth_array)
#         depth_masked = np.where(mask, depth_array, np.nan)
        
#         # ==========================================
#         # 🌟 稀疏点增强处理 (形态学膨胀)
#         # ==========================================
#         if DILATE_SPARSE:
#             # 创建膨胀核 (矩形核，也可以用 MORPH_ELLIPSE 画圆点)
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_KERNEL, DILATE_KERNEL))
            
#             # 深度膨胀: 先把 NaN 补 0 进行膨胀，最大池化会让深度值扩散
#             temp_depth = np.nan_to_num(depth_masked, nan=0.0)
#             dilated_depth = cv2.dilate(temp_depth, kernel)
            
#             # 掩码膨胀: 确定扩散后的有效区域
#             mask_uint8 = mask.astype(np.uint8)
#             dilated_mask = cv2.dilate(mask_uint8, kernel) > 0
            
#             # 重新赋值：将被膨胀放大的区域填回原本或扩散后的深度值，其余填 NaN
#             depth_masked = np.where(dilated_mask, dilated_depth, np.nan)
#             mask = dilated_mask # 更新掩码供后续探针寻找使用

#         if np.isnan(depth_masked).all():
#             print(f"⚠️ {stem} 全是无效深度值，跳过。")
#             return
            
#         # 2. 计算色阶
#         if vmin is None: vmin = np.nanquantile(depth_masked, 0.01)
#         if vmax is None: vmax = np.nanquantile(depth_masked, 0.99)

#         # 准备输出路径
#         path_pure = out_dir / f"{stem}_1_pure.png"
#         path_cbar = out_dir / f"{stem}_2_cbar.png"
#         path_probes = out_dir / f"{stem}_3_probes.png"

#         h, w = depth_array.shape
#         aspect = w / h
#         fig_w = max(6, 5 * aspect)

#         # ==========================================
#         # 🎨 输出 1: 纯粹的深度图 (1:1 原生像素导出)
#         # ==========================================
#         plt.imsave(str(path_pure), depth_masked, cmap=self.cmap, vmin=vmin, vmax=vmax)

#         # ==========================================
#         # 🎨 输出 2: 深度图 + Colorbar
#         # ==========================================
#         fig, ax = plt.subplots(figsize=(fig_w, 5))
#         im = ax.imshow(depth_masked, cmap=self.cmap, vmin=vmin, vmax=vmax)
#         ax.axis('off')
        
#         cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         cbar.set_label('Absolute Depth (m)', fontsize=14, fontweight='bold', labelpad=12)
#         cbar.ax.tick_params(labelsize=12)
        
#         plt.tight_layout()
#         fig.savefig(path_cbar, bbox_inches='tight', dpi=200)
#         plt.close(fig)

#         # ==========================================
#         # 🎨 输出 3: 深度图 + 探针探测值 (智能中心吸附版)
#         # ==========================================
#         fig, ax = plt.subplots(figsize=(fig_w, 5))
#         ax.imshow(depth_masked, cmap=self.cmap, vmin=vmin, vmax=vmax)
#         ax.axis('off')

#         # --- 🤖 智能寻找有效探针点 ---
#         def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
#             if valid_mask[tgt_y, tgt_x]:
#                 return (tgt_x, tgt_y)
            
#             ys, xs = np.where(valid_mask)
#             if len(ys) == 0:
#                 return None 
                
#             dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
#             best_idx = np.argmin(dist_sq)
#             return (xs[best_idx], ys[best_idx])

#         # 🔧 1. 设定理想点位置
#         cx, cy = w // 2, h // 2       
#         offset_x = w // 4             

#         pt1 = get_nearest_valid_point(cx - offset_x, cy, mask)
#         pt2 = get_nearest_valid_point(cx + offset_x, cy, mask)
        
#         test_points = [pt for pt in [pt1, pt2] if pt is not None]

#         # --- 🎨 绘制点和悬浮文字 ---
#         colors = ['#FF3366', '#00E676'] # 亮粉和亮绿
        
#         for pt_idx, (x, y) in enumerate(test_points):
#             c = colors[pt_idx % len(colors)]
            
#             ax.plot(x, y, marker='o', markerfacecolor=c, markeredgecolor='white', 
#                     markersize=14, markeredgewidth=2.5, zorder=5)
            
#             val = depth_masked[y, x]
#             text_str = f"{val:.1f}m"
            
#             y_offset = -60 if pt_idx == 0 else 40 
            
#             ax.text(
#                 x, y + y_offset, 
#                 text_str,
#                 color='black', fontsize=15, fontweight='bold',
#                 ha='center', va='center', zorder=6,
#                 bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor="none", alpha=0.85)
#             )

#         plt.tight_layout()
#         fig.savefig(path_probes, bbox_inches='tight', dpi=200, facecolor='white')
#         plt.close(fig)

# # ================= 3. 文件解析与批处理 =================

# def resolve_npy_paths(targets):
#     resolved = []
#     if isinstance(targets, (str, Path)):
#         targets = [targets]
#     for t in targets:
#         p = Path(t).resolve()
#         if p.is_file() and p.suffix == '.npy':
#             resolved.append(p)
#         elif p.is_dir():
#             resolved.extend(sorted(p.rglob('*.npy')))
#     return list(dict.fromkeys(resolved))

# if __name__ == "__main__":
#     print(f"{'='*60}\n📊 深度图三重渲染器 (支持稀疏点增强与白底)\n{'='*60}")
    
#     target_paths = resolve_npy_paths(INPUT_TARGETS)
#     if not target_paths:
#         print("❌ 未找到任何 .npy 文件。")
#         sys.exit(1)
        
#     out_dir = Path(OUTPUT_DIR)
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     renderer = DepthRenderer()
    
#     for npy_path in tqdm(target_paths, desc="Rendering Trios"):
#         try:
#             data = np.load(npy_path)
            
#             if data.ndim == 3 and data.shape[0] == 1: data = data[0]
#             elif data.ndim == 3 and data.shape[2] == 1: data = data[:, :, 0]
                
#             if data.ndim == 3 and data.shape[2] == 3:
#                 continue
                
#             parent_name = npy_path.parent.name
#             if parent_name == npy_path.parent.parent.name or "out" in parent_name.lower():
#                 stem = npy_path.stem
#             else:
#                 stem = f"{parent_name}_{npy_path.stem}"
                
#             renderer.render_trio(
#                 depth_array=data,
#                 stem=stem,
#                 out_dir=out_dir,
#                 vmin=GLOBAL_VMIN,
#                 vmax=GLOBAL_VMAX
#             )
            
#         except Exception as e:
#             import traceback
#             print(f"\n❌ 处理失败 {npy_path.name}: {e}")
#             traceback.print_exc()

#     print(f"\n✅ 渲染完成！所有图片已保存至: {out_dir}")