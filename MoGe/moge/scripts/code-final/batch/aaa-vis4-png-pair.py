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


    
    # ("/data1/szq/Val/visfigure3/3e72d469f997268640d28dfdbeb46f132163a187/image.jpg", "/data1/szq/Val/visfigure3/3e72d469f997268640d28dfdbeb46f132163a187/depth.npy"),    # 稠密的 预测值1
    # ("/data1/szq/Val/visfigure3/5abdbd8d5ebae9b8f6fe2b85e0e8567465064220/image.jpg", "/data1/szq/Val/visfigure3/5abdbd8d5ebae9b8f6fe2b85e0e8567465064220/facdepth.npy"),  # 稠密的 预测值2
    # ("/data1/szq/Val/visfigure3/14f5f6d4a4b095a3d826d0ab3aacc58e30e122ad/image.jpg", "/data1/szq/Val/visfigure3/14f5f6d4a4b095a3d826d0ab3aacc58e30e122ad/ladndepth.npy"),        
    # ("/data1/szq/Val/visfigure3/b0bbd654bf68e409d686bae27f498c1e42da0df2/image.jpg", "/data1/szq/Val/visfigure3/b0bbd654bf68e409d686bae27f498c1e42da0df2/camdepth.npy"),    
    # ("/data1/szq/Val/visfigure3/d806fb9af1e601ff626095028f49b444d7eab758/image.jpg", "/data1/szq/Val/visfigure3/d806fb9af1e601ff626095028f49b444d7eab758/parkdepth.npy"),    
    # ("/data1/szq/Val/visfigure3/1768577199971_6 copy.png", "/data1/szq/Val/visfigure3/1768577199971_6.npy"),        
    # ("/data1/szq/Val/visfigure3/1770914227171_20 copy 2.png", "/data1/szq/Val/visfigure3/1770914227171_20 copy.npy"), 

    # ("/data1/szq/Val/visfigure3-oblique/smbu000025.jpg", "/data1/szq/Val/visfigure3-oblique/000025.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/sziit000048.jpg", "/data1/szq/Val/visfigure3-oblique/000048.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/sztu000095.jpg", "/data1/szq/Val/visfigure3-oblique/000095.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/hav000099.jpg", "/data1/szq/Val/visfigure3-oblique/000099.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/lfls000245.jpg", "/data1/szq/Val/visfigure3-oblique/000245.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/upper000340.jpg", "/data1/szq/Val/visfigure3-oblique/000340.npy"),    # 备注内容

    # ("/data1/szq/Val/visfigure3-oblique/1658137447.624904733.jpg", "/data1/szq/Val/visfigure3-oblique/1658137447.624904733_depth.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/1658228393.459161661.jpg", "/data1/szq/Val/visfigure3-oblique/1658228393.459161661_depth.npy"),    # 备注内容
    # ("/data1/szq/Val/visfigure3-oblique/airport1698217792.099807978.jpg", "/data1/szq/Val/visfigure3-oblique/1669702110.399817944_depth.npy"),    # 备注内容
    # ("example_rgb.jpg", "/data1/szq/Val/visfigure3-oblique/1698217792.099807978_depth.npy"), 
    
    # ("/data1/szq/Val/visfig-oblique2/74cd1184d0a26a13ba1bca5708befbe41def6a2e.jpg", "/data1/szq/Val/visfig-oblique2/74cd1184d0a26a13ba1bca5708befbe41def6a2e.npy"), 
    #       ("/data1/szq/Val/visfig-oblique2/84d73afeadd4c39278d172f3810039b2b365226a.jpg", "/data1/szq/Val/visfig-oblique2/84d73afeadd4c39278d172f3810039b2b365226a.npy"), 
            #   ("/data1/szq/Val/visfig-oblique2/self/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.jpg", "/data1/szq/Val/visfig-oblique2/self/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy"), 
                        #   ("/data1/szq/Val/Vis/visfig-oblique2/self/0cb7f8b6503bde1d9b804f1bbed0dc15519e94e2.jpg", "/data1/szq/Val/Vis/visfig-oblique2/self/0cb7f8b6503bde1d9b804f1bbed0dc15519e94e2.npy"), 
                        #        ("/data1/szq/Val/Vis/visfig-oblique2/self/0d51caf860c6612ab488e7dc698ae923294b1d3d.jpg", "/data1/szq/Val/Vis/visfig-oblique2/self/0d51caf860c6612ab488e7dc698ae923294b1d3d.npy"), 
                        #                 ("/data1/szq/Val/Vis/visfig-oblique2/self/1d7a8cc1ebfbd9425c7cd5ea1b8a246317ed3725.jpg", "/data1/szq/Val/Vis/visfig-oblique2/self/1d7a8cc1ebfbd9425c7cd5ea1b8a246317ed3725.npy"), 
                        # ("/data1/szq/Val/Vis/visfig-oblique2/self/4c62569bbee763281508e3aea502f57b40a19829.jpg", "/data1/szq/Val/Vis/visfig-oblique2/self/4c62569bbee763281508e3aea502f57b40a19829.npy"), 
                            #    ("/data1/szq/Val/Vis/visfig-oblique2/self/5d2078ab86bf3fa29fdbcc21a1229aa478b79eef.jpg", "/data1/szq/Val/Vis/visfig-oblique2/self/5d2078ab86bf3fa29fdbcc21a1229aa478b79eef.npy"), 
                                        # ("/data1/szq/Val/Vis/visfig-oblique2/self/1ddfb001659b9cfe2e10a82b66ec1c065901973f.jpg", "/data1/szq/Val/Vis/visfig-oblique2/self/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy"), 


#  ("/data1/szq/Val/Vis/visfig-oblique2/Wild/Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_s008_f002092.jpg", "/data1/szq/Val/Vis/visfig-oblique2/Wild/Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_s008_f002092.npy"), 
#  ("/data1/szq/Val/Vis/visfig-oblique2/Wild/Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_s018_f005391.jpg", "/data1/szq/Val/Vis/visfig-oblique2/Wild/Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_s018_f005391.npy"), 
#  ("/data1/szq/Val/Vis/visfig-oblique2/Wild/TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_s249_f023895.jpg", "/data1/szq/Val/Vis/visfig-oblique2/Wild/TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_s249_f023880.npy"), 
#                ("/data1/szq/Val/visfig-oblique2/aligned_outputwild/scene_039_frame_008443.jpg", "/data1/szq/Val/visfig-oblique2/aligned_outputwild/scene_039_frame_008443.npy"), 
# ("/data1/szq/Val/visfig-oblique2/aligned_outputwild/scene_019_frame_003390.jpg", "/data1/szq/Val/visfig-oblique2/aligned_outputwild/scene_019_frame_003390.npy"), 
#               ("/data1/szq/Val/visfig-oblique2/aligned_outputwild/Orbit_Shot_of_The_Roman_Colosseum_4K_Free_Download_Amazing_Arial_View_Nlw5AfgKpQg_s000_f000000.jpg", "/data1/szq/Val/visfig-oblique2/aligned_outputwild/Orbit_Shot_of_The_Roman_Colosseum_4K_Free_Download_Amazing_Arial_View_Nlw5AfgKpQg_s000_f000000.npy"), 
#               ("/data1/szq/Val/visfig-oblique2/aligned_outputwild/Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_s041_f012529.jpg", "/data1/szq/Val/visfig-oblique2/aligned_outputwild/Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_s041_f012529.npy"), 
#               ("/data1/szq/Val/visfig-oblique2/aligned_outputwild/Anfield_Stadium_Development_Liverpool_FC_Phase_2_Anfield_Road_March_2022_Drone_Footage_D0H2_Z6uDB4_s002_f003364.jpg", "/data1/szq/Val/visfig-oblique2/aligned_outputwild/Anfield_Stadium_Development_Liverpool_FC_Phase_2_Anfield_Road_March_2022_Drone_Footage_D0H2_Z6uDB4_s002_f003364.npy"), 
            
            #       ("/data1/szq/Val/visfig-oblique2/74cd1184d0a26a13ba1bca5708befbe41def6a2e.jpg", "/data1/szq/Val/visfig-oblique2/74cd1184d0a26a13ba1bca5708befbe41def6a2e.npy"),  # 备注内容

#  ("/data1/szq/Val/Vis/visfig-oblique2/oblique/4ac7ce3fafcdab8fdc4321be52093b96866ad5c8.jpg", "/data1/szq/Val/Vis/visfig-oblique2/oblique/4ac7ce3fafcdab8fdc4321be52093b96866ad5c8.npy"), 
#  ("/data1/szq/Val/Vis/visfig-oblique2/oblique/79b23c259a5094eaa4c7f5d614ce532dffbf2bf3.jpg", "/data1/szq/Val/Vis/visfig-oblique2/oblique/79b23c259a5094eaa4c7f5d614ce532dffbf2bf3.npy"), 
#   ("/data1/szq/Val/Vis/visfig-oblique2/oblique/79c2b8a24b49b31e0d50061b77438c4accf6f449.jpg", "/data1/szq/Val/Vis/visfig-oblique2/oblique/79c2b8a24b49b31e0d50061b77438c4accf6f449.npy"), 
#    ("/data1/szq/Val/Vis/visfig-oblique2/oblique/000084.jpg", "/data1/szq/Val/Vis/visfig-oblique2/oblique/000084.npy"), 
#  ("/data1/szq/Val/Vis/visfig-oblique2/oblique/b74f1e66f12114088a9aec64dd8414f1fd9bdffb.jpg", "/data1/szq/Val/Vis/visfig-oblique2/oblique/b74f1e66f12114088a9aec64dd8414f1fd9bdffb.npy"), 
#   ("/data1/szq/Val/visfig-oblique2/aligned_outputoblique/500c24ef6d585d71937b75a841b59201a049f7d2.jpg", "/data1/szq/Val/visfig-oblique2/aligned_outputoblique/500c24ef6d585d71937b75a841b59201a049f7d2.npy"), 


#   ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/6421cede38b0a8abae54ee3037bd52746707dcd1.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/6421cede38b0a8abae54ee3037bd52746707dcd1.npy"), 
#   a12fd8163e839a22f340525e6f222e7150cba078
# e607fd7d31b36a74e405626959c764ddec1b841c
# 098ed6445d12adbfa82d18b374a56a1d1402c71a

# 5d8316b05cf61fd102584274ef14c4dbb88d5fef
# 92c170e95a5e3948e912437a3bfe03e7a6cb6a7b
# 74344880b295d15334852916152850aa9a5269f5
# 116c89c771990d79796409ddfb3e6ce8f1611418

# 12feed6c879d71cbfe615af29394095c8375ef0e
# 1e337274b869243edde6ba81f88a0cd75652edf0
# 24089b132e7df0a92d5b78e09a9eec249b3b64c8
# ea7090c07c037e453ddc6d1036109120689e49ad

# faac9d8272bb932d753a5c2f37ae3c047b30c7a1
# eac953c5232fbb2dcf9bacaeecf1a85e72d66e56
# 09c8fbab65562d1c004ab0f0a8f433f830e3b0c4
# 603c8d25e50a68d211a0e511e1ac6e179ec531aa
# ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/6421cede38b0a8abae54ee3037bd52746707dcd1.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/6421cede38b0a8abae54ee3037bd52746707dcd1.npy"),
    
#     # 第一组
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/a12fd8163e839a22f340525e6f222e7150cba078.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/a12fd8163e839a22f340525e6f222e7150cba078.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/e607fd7d31b36a74e405626959c764ddec1b841c.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/e607fd7d31b36a74e405626959c764ddec1b841c.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/098ed6445d12adbfa82d18b374a56a1d1402c71a.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/098ed6445d12adbfa82d18b374a56a1d1402c71a.npy"),
    
#     # 第二组
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/5d8316b05cf61fd102584274ef14c4dbb88d5fef.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/5d8316b05cf61fd102584274ef14c4dbb88d5fef.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/92c170e95a5e3948e912437a3bfe03e7a6cb6a7b.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/92c170e95a5e3948e912437a3bfe03e7a6cb6a7b.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/74344880b295d15334852916152850aa9a5269f5.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/74344880b295d15334852916152850aa9a5269f5.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/116c89c771990d79796409ddfb3e6ce8f1611418.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/116c89c771990d79796409ddfb3e6ce8f1611418.npy"),
    
#     # 第三组
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/12feed6c879d71cbfe615af29394095c8375ef0e.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/12feed6c879d71cbfe615af29394095c8375ef0e.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/1e337274b869243edde6ba81f88a0cd75652edf0.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/1e337274b869243edde6ba81f88a0cd75652edf0.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/24089b132e7df0a92d5b78e09a9eec249b3b64c8.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/24089b132e7df0a92d5b78e09a9eec249b3b64c8.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/ea7090c07c037e453ddc6d1036109120689e49ad.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/ea7090c07c037e453ddc6d1036109120689e49ad.npy"),
    
#     # 第四组
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/faac9d8272bb932d753a5c2f37ae3c047b30c7a1.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/faac9d8272bb932d753a5c2f37ae3c047b30c7a1.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/eac953c5232fbb2dcf9bacaeecf1a85e72d66e56.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/eac953c5232fbb2dcf9bacaeecf1a85e72d66e56.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/09c8fbab65562d1c004ab0f0a8f433f830e3b0c4.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/09c8fbab65562d1c004ab0f0a8f433f830e3b0c4.npy"),
#     ("/data1/szq/Val/Bench/Cleaned_Dataset_Factory/image/603c8d25e50a68d211a0e511e1ac6e179ec531aa.jpg", "/data1/szq/Val/Bench/Cleaned_Dataset_Factory/depth/603c8d25e50a68d211a0e511e1ac6e179ec531aa.npy"),


           
]

OUTPUT_DIR = "/data1/szq/Val/visfig-oblique2/aligned_output-self" 

COLORMAP = 'Spectral'       
INVALID_COLOR = 'black'     

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

    def colorize_depth_to_rgb(self, depth_array, mask, vmin, vmax):
        """将单通道深度图直接映射为 3通道 RGB 数组，便于与原图数组完美拼接"""
        norm_depth = (depth_array - vmin) / (vmax - vmin + 1e-8)
        norm_depth = np.clip(norm_depth, 0, 1)
        
        # 应用 cmap 得到 RGBA, 取前3个通道
        colored = self.cmap(norm_depth)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        
        # 填充无效区域
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
        
        # 1. 数组级别色彩映射
        depth_colored = self.colorize_depth_to_rgb(depth_array, mask, local_vmin, local_vmax)
        
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

        # 在切分处画一条白色的分割线（如果不想要，可以把这行注释掉）
        ax.axvline(x=mid_w, color='white', linewidth=2, zorder=3)

        # 4. 计算本张图片专属的单个探针
        # 目标位置：右半区的中心，即宽度 3/4 处，高度 1/2 处
        target_x = int(w_d * 0.75)
        target_y = h_d // 2
        
        # 获取最接近该目标点的有效深度坐标
        probe_pt = get_nearest_valid_point(target_x, target_y, mask)

        if probe_pt is not None:
            px, py = probe_pt
            
            # 确保找到的点确实在右半边（防止右半边全是无效值，被迫取到左半边的情况）
            if px >= mid_w:
                val = depth_array[py, px]
                c = '#FF3366'  # 使用高亮粉红色
                
                # 直接在坐标 (px, py) 绘制单个探针
                ax.plot(px, py, marker='o', markerfacecolor=c, markeredgecolor='white', 
                        markersize=14, markeredgewidth=2.5, zorder=5)
                
                # 添加深度数值标签
                text_str = f"{val:.1f}m"
                ax.text(
                    px, py - 40,  # 标签固定显示在探针上方一点
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
            return data.astype(np.float32) / 256.0
            
    except Exception as e:
        print(f"❌ 读取 深度图 异常 {file_path}: {e}")
    return None

if __name__ == "__main__":
    print(f"{'='*60}\n📊 RGB-Depth 1:1对半分切拼接渲染器 (自适应色阶 & 单探针)\n{'='*60}")
    
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
                print(f"⚠️ {depth_path.stem} 尺寸不一致，将 深度图({w_d}x{h_d}) 上采样至 RGB原图尺寸({w_r}x{h_r})")
                # 深度图上采样推荐使用双线性插值(INTER_LINEAR)或最近邻(INTER_NEAREST)
                depth_data = cv2.resize(depth_data, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
        else:
            # 兼容：如果没配置RGB，生成一个纯黑的占位图保证流程不中断
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
