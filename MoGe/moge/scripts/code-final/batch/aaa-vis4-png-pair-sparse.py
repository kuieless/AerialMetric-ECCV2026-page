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

# 修改为元组列表：(RGB原图路径, 深度图路径)
INPUT_PAIRS = [
    ("/data1/szq/Val/visfigure3-oblique/1658137447.624904733.jpg", "/data1/szq/Val/visfigure3-oblique/1658137447.624904733_depth.npy"),    # 备注内容
    ("/data1/szq/Val/visfigure3-oblique/1658228393.459161661.jpg", "/data1/szq/Val/visfigure3-oblique/1658228393.459161661_depth.npy"),    # 备注内容
    ("/data1/szq/Val/visfigure3-oblique/airport1698217792.099807978.jpg", "/data1/szq/Val/visfigure3-oblique/1698217792.099807978_depth.npy"),    # 备注内容
    # ("example_rgb.jpg", "/data1/szq/Val/visfigure3-oblique/1698217792.099807978_depth.npy"),    # 备注内容
]

OUTPUT_DIR = "/data1/szq/Val/visfigure3-oblique/aligned_outputsparse2" 

COLORMAP = 'Spectral'       
INVALID_COLOR = 'black' 

# 👇 新增参数：稀疏点视觉放大倍数
POINT_SIZE = 5   # 设为 1 表示不放大；对于稀疏深度图，建议设为 3, 5 或 7

# ================= 2. 核心渲染器 & 工具 =================

def get_valid_mask(depth_array: np.ndarray) -> np.ndarray:
    """
    健壮的掩码生成器：严格剔除 NaN, Inf, <=0 的值，以及极其离谱的噪点(>1000m)
    """
    return np.isfinite(depth_array) & (depth_array > 1e-3) & (depth_array < 1000.0)

def get_nearest_valid_point(tgt_x, tgt_y, valid_mask):
    if valid_mask[tgt_y, tgt_x]:
        return (tgt_x, tgt_y)
    
    ys, xs = np.where(valid_mask)
    if len(ys) == 0:
        return None 
        
    dist_sq = (xs - tgt_x)**2 + (ys - tgt_y)**2
    best_idx = np.argmin(dist_sq)
    return (int(xs[best_idx]), int(ys[best_idx]))

class AlignedPairRenderer:
    def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
        self.cmap = plt.get_cmap(cmap_name).copy()
        # 将无效颜色转换为 RGB 的 uint8 格式
        self.invalid_color_rgb = tuple(int(c * 255) for c in mcolors.to_rgb(invalid_color))

    def colorize_depth_to_rgb(self, depth_array, mask, vmin, vmax, point_size=1):
        """将单通道深度图直接映射为 3通道 RGB 数组，便于与原图数组完美拼接"""
        norm_depth = (depth_array - vmin) / (vmax - vmin + 1e-8)
        norm_depth = np.clip(norm_depth, 0, 1)
        
        # 应用 cmap 得到 RGBA, 取前3个通道
        colored = self.cmap(norm_depth)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        
        # 🌟 视觉膨胀处理（解决稀疏点看不清的问题）
        if point_size > 1:
            # 1. 提取有效颜色，把背景置为纯黑（防止灰色背景被膨胀到颜色里）
            valid_colored = colored.copy()
            valid_colored[~mask] = 0
            
            # 2. 生成圆形膨胀核
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (point_size, point_size))
            
            # 3. 对颜色和掩码同时进行膨胀
            colored_dilated = cv2.dilate(valid_colored, kernel)
            mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            
            # 4. 把膨胀后的背景填回无效颜色
            colored_dilated[~mask_dilated] = self.invalid_color_rgb
            return colored_dilated
        else:
            # 如果是稠密深度，或者 point_size == 1，按原样填充
            colored[~mask] = self.invalid_color_rgb
            return colored

    def render_pair(self, rgb_array: np.ndarray, depth_array: np.ndarray, stem: str, out_dir: Path):
        mask = get_valid_mask(depth_array)
        
        if not np.any(mask):
            print(f"⚠️ {stem} 全是无效深度值或被全部过滤，跳过。")
            return

        # 动态计算当前图像的局部色阶 (剔除 1% 极端噪点)
        valid_pixels = depth_array[mask]
        local_vmin = np.percentile(valid_pixels, 1)
        local_vmax = np.percentile(valid_pixels, 99)
        if local_vmax - local_vmin < 1e-3:
            local_vmax = local_vmin + 1.0

        h_d, w_d = depth_array.shape
        mid_w = w_d // 2  # 找到图像的垂直中线
        
        # 1. 数组级别色彩映射 (传入 point_size 进行视觉加粗)
        depth_colored = self.colorize_depth_to_rgb(depth_array, mask, local_vmin, local_vmax, point_size=POINT_SIZE)
        
        # 2. 从中间切开拼接：左半边原图，右半边深度图（保持原本分辨率）
        canvas = np.zeros((h_d, w_d, 3), dtype=np.uint8)
        canvas[:, :mid_w] = rgb_array[:, :mid_w]      # 左边填入 RGB 原图的左半边
        canvas[:, mid_w:] = depth_colored[:, mid_w:]  # 右边填入 Depth 的右半边

        # 3. 创建 matplotlib 画布仅仅为了绘制文字和探针
        fig_w = max(10, 8 * (w_d / h_d))
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * (h_d / w_d)))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # 消除所有内边距
        
        ax.imshow(canvas)
        ax.axis('off')

        # 在切分处画一条白色的分割线
        ax.axvline(x=mid_w, color='white', linewidth=2, zorder=3)

        # 4. 计算本张图片专属的单个探针 (注意：探针寻找依然基于严格的原始 mask，而不是膨胀后的 mask)
        target_x = int(w_d * 0.75)
        target_y = h_d // 2
        
        probe_pt = get_nearest_valid_point(target_x, target_y, mask)

        if probe_pt is not None:
            px, py = probe_pt
            
            if px >= mid_w:
                val = depth_array[py, px]
                c = '#FF3366'  
                
                ax.plot(px, py, marker='o', markerfacecolor=c, markeredgecolor='white', 
                        markersize=14, markeredgewidth=2.5, zorder=5)
                
                text_str = f"{val:.1f}m"
                ax.text(
                    px, py - 40,  
                    text_str,
                    color='black', fontsize=15, fontweight='bold',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor=c, linewidth=2, alpha=0.9)
                )

        out_path = out_dir / f"{stem}_split_view.png"
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)

# ================= 3. 数据加载与处理 =================

def load_rgb_data(file_path: Path):
    try:
        img = cv2.imread(str(file_path))
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换为 RGB 以适配 Matplotlib
    except Exception as e:
        print(f"❌ 读取 RGB 异常 {file_path}: {e}")
        return None

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
            return data.astype(np.float32) / 5000.0
            
    except Exception as e:
        print(f"❌ 读取 深度图 异常 {file_path}: {e}")
    return None

if __name__ == "__main__":
    print(f"{'='*60}\n📊 RGB-Depth 1:1对半分切拼接渲染器 (稀疏点加粗版)\n{'='*60}")
    
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 正在加载数据并核对尺寸...")
    
    loaded_datasets = []
    
    for rgb_str, depth_str in INPUT_PAIRS:
        rgb_path, depth_path = Path(rgb_str), Path(depth_str)
        
        if not depth_path.exists():
            print(f"❌ 找不到深度文件，跳过: {depth_path.name}")
            continue
            
        rgb_data = load_rgb_data(rgb_path)
        depth_data = load_depth_data(depth_path)
        
        if depth_data is None: continue
        
        # --- 尺寸对齐保护机制 ---
        if rgb_data is not None:
            h_d, w_d = depth_data.shape
            h_r, w_r = rgb_data.shape[:2]
            if (h_d, w_d) != (h_r, w_r):
                print(f"⚠️ {depth_path.stem} 尺寸不一致，将 RGB({w_r}x{h_r}) 强制缩放至深度图尺寸({w_d}x{h_d})")
                rgb_data = cv2.resize(rgb_data, (w_d, h_d), interpolation=cv2.INTER_CUBIC)
        else:
            rgb_data = np.zeros((*depth_data.shape, 3), dtype=np.uint8)
            
        loaded_datasets.append((depth_path, rgb_data, depth_data))

    if not loaded_datasets:
        print("❌ 没有成功加载任何有效数据对。")
        sys.exit(1)

    # --- 开始渲染 ---
    renderer = AlignedPairRenderer()
    
    for file_path, rgb_data, depth_data in tqdm(loaded_datasets, desc="Rendering"):
        stem = file_path.stem
        renderer.render_pair(
            rgb_array=rgb_data,
            depth_array=depth_data,
            stem=stem,
            out_dir=out_dir
        )
            
    print(f"\n✅ 渲染完成！所有切分视图已保存至: {out_dir}")