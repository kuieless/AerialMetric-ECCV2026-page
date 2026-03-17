import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 防止在无界面服务器报错
import matplotlib.pyplot as plt

# ================= 1. 用户配置区域 =================
TARGET_SIZE = (1280, 960) # (宽, 高)
INPUT_TARGETS = [
    # "/data1/szq/Val/visfigure3/3e72d469f997268640d28dfdbeb46f132163a187/depth.npy",    # 稠密的 预测值1
    # "/data1/szq/Val/visfigure3/5abdbd8d5ebae9b8f6fe2b85e0e8567465064220/facdepth.npy",  # 稠密的 预测值2
    # "/data1/szq/Val/visfigure3/14f5f6d4a4b095a3d826d0ab3aacc58e30e122ad/ladndepth.npy",         
    # "/data1/szq/Val/visfigure3/b0bbd654bf68e409d686bae27f498c1e42da0df2/camdepth.npy",     
    # "/data1/szq/Val/visfigure3/d806fb9af1e601ff626095028f49b444d7eab758/parkdepth.npy",     
    # "/data1/szq/Val/visfigure3/1768577272214_7.npy",        
    #     "/data1/szq/Val/visfigure3/1770914227171_20 copy.npy", 

    # "/data1/szq/Val/visfigure3-oblique/000025.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/000048.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/000095.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/000099.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/000245.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/000340.npy",    # 备注内容

    # "/data1/szq/Val/visfigure3-oblique/1658137447.624904733_depth.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/1658228393.459161661_depth.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/1669702110.399817944_depth.npy",    # 备注内容
    # "/data1/szq/Val/visfigure3-oblique/1698217792.099807978_depth.npy",    # 备注内容                 # 稀疏的 KITTI 真值
# "/data1/yzy/Benchmark/Metric3D/data/kitti_demo/depth/0000000005.png", 
# "/data1/szq/Val/vissss/kitti/Baseline/0000000005/depth.npy", 
# "/data1/szq/Val/vissss/kitti/LoRA_Rank96/0000000005/loradepth.npy", 

# "/data1/yzy/Benchmark/Metric3D/data/kitti_demo/rgb/0000000050.png", 
# "/data1/szq/Val/vissss/kitti/Baseline/0000000050/depth.npy", 
# "/data1/szq/Val/vissss/kitti/LoRA_Rank96/0000000050/loradepth.npy", 

# "/data1/yzy/Benchmark/Metric3D/data/kitti_demo/rgb/0000000100.png", 
# "/data1/szq/Val/vissss/kitti/Baseline/0000000100/depth.npy", 
# "/data1/szq/Val/vissss/kitti/LoRA_Rank96/0000000100/loradepth.npy", 


"/data1/yzy/Benchmark/Metric3D/data/nyu_demo/depth/sync_depth_00000.png", 
"/data1/szq/Val/Viss/vissss/nyu/Baseline/rgb_00000/depth.npy", 
"/data1/szq/Val/Viss/vissss/nyu/LoRA_Rank96/rgb_00000/loradepth.npy", 

# "/data1/yzy/Benchmark/Metric3D/data/nyu_demo/depth/sync_depth_00050.png", 
# "/data1/szq/Val/vissss/nyu/Baseline/rgb_00050/depth.npy", 
# "/data1/szq/Val/vissss/nyu/LoRA_Rank96/rgb_00050/loradepth.npy", 

# "/data1/yzy/Benchmark/Metric3D/data/nyu_demo/depth/sync_depth_00100.png", 
# "/data1/szq/Val/vissss/nyu/Baseline/rgb_00100/depth.npy", 
# "/data1/szq/Val/vissss/nyu/LoRA_Rank96/rgb_00100/loradepth.npy", 


# "/data1/szq/Val/Viss/Vis-sota/pic-npy/000025.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/000025.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/pro000025.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/000025.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/000025/depth.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/000025/depth.npy", 

# "/data1/szq/Val/Viss/Vis-sota/pic-npy/000099.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/000099.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/000099.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/000099.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/000099/depth.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/000099/depth.npy", 

# "/data1/szq/Val/Viss/Vis-sota/pic-npy/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b/depth.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/1e4da0955364900ce9925ab90c4ecdc1a2b74e5b/depth.npy", 


# "/data1/szq/Val/Viss/Vis-sota/pic-npy/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/zoedepth/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/depth_pro_results/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/uni/1ddfb001659b9cfe2e10a82b66ec1c065901973f.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/Baseline/1ddfb001659b9cfe2e10a82b66ec1c065901973f/depth.npy", 
# "/data1/szq/Val/Viss/Vis-sota/infer-photo/moge2/LoRA_Rank96/1ddfb001659b9cfe2e10a82b66ec1c065901973f/depth.npy", 


# "/data1/yzy/Benchmark/Metric3D/data/nyu_demo/depth/sync_depth_00000.png", 
# "/data1/szq/Val/Viss/Vis-sota/FInal/ground/zoeimage.npy", 
# "/data1/szq/Val/Viss/Vis-sota/FInal/ground/proimage.npy", 
# "/data1/szq/Val/Viss/Vis-sota/FInal/ground/image.npy", 
# "/data1/szq/Val/Viss/vissss/nyu/Baseline/rgb_00000/depth.npy", 
# "/data1/szq/Val/Viss/vissss/nyu/LoRA_Rank96/rgb_00000/loradepth.npy", 

]



OUTPUT_DIR = "/data1/szq/Val/Viss/Vis-sota/FInal3/figground-1" 
MANUAL_PROBES = [
    # (320, 480),  # 左侧中景
    # # (960, 480),  # 右侧中景
    # # (640, 320),  # 上部远景
    # # (640, 720),  # 下部近景
    # # (400, 600),  # 左下
    # (880, 400)   # 右上

    # --- 你原本保留的点 (已按 640x480 缩放) ---

    # --- 🎯 绝对C位：正前方深度测距 ---
    # (320, 240),  # 0: 画面绝对正中心（核心障碍物）
    # (320, 160),  # 1: 正中偏远（消失点、远处建筑或天空交界）
    # (320, 340),  # 2: 正中偏近（正前方的路面或地板）

    # # --- 🛡️ 内圈护卫：测试主体结构的连贯性 ---
    # (240, 210),  # 3: 左中远（左侧远处建筑/车辆）
    (400, 210),  # 4: 右中远（右侧远处建筑/车辆）
    # (240, 290),  # 5: 左中近（左侧近处路面/车道线）
    # (400, 290),  # 6: 右中近（右侧近处路面/车道线）


    # # --- 🎯 之前的中心和左侧核心点（精简保留几个对比用） ---
    # (320, 240),  # 画面绝对正中心
    # (320, 340),  # 正中偏下（路面）
    # (160, 240),  # 左侧对照组
    
    # ==========================================================
    # 🚗 疯狂爆破区：极右侧车/墙体密集扫描网 (X轴 500~620)
    # ==========================================================
    
    # # 1. 锚点列 (X=500)
    # (500, 200),  # 锚点正上
    # (500, 240),  # 🎯 原锚点
    # (500, 280),  # 锚点正下

    # # 2. 向右推进第一层 (X=530)
    # (530, 190),  # 右1 + 上
    # (530, 240),  # 右1 + 中
    (530, 290),  # 右1 + 下

    # # 3. 向右推进第二层 (X=560)
    # (560, 180),  # 右2 + 上
    # (560, 240),  # 右2 + 中
    # (560, 300),  # 右2 + 下




]
USE_MANUAL_PROBES = True
# 如果你想恢复自动模式，可以设为 None
# USE_MANUAL_PROBES = True
COLORMAP = 'Spectral'       
GLOBAL_VMIN = None          
GLOBAL_VMAX = None          
INVALID_COLOR = 'white'     

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


class DepthRenderer:
    def __init__(self, cmap_name=COLORMAP, invalid_color=INVALID_COLOR):
        self.cmap = plt.get_cmap(cmap_name).copy()
        self.cmap.set_bad(color=invalid_color) 

    def render_trio(self, depth_array: np.ndarray, stem: str, out_dir: Path, target_probes: list, vmin: float, vmax: float):
        # 1. 提取有效掩码
        mask = get_valid_mask(depth_array)
        if not np.any(mask):
            print(f"⚠️ {stem} 全是无效深度值，跳过。")
            return
            
        # 2. 强制尺寸对齐 (关键：确保排版一致)
        # 注意：cv2.resize 接收的是 (宽, 高)
        depth_resized = cv2.resize(depth_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        
        # 3. 计算局部量程 (1-99%)
        display_vmin = vmin
        display_vmax = vmax

        # 如果深度值超出了全局范围，超出部分会被 clip 到边界颜色，从而在图上呈现极端的红色或蓝色
        # 4. 数据裁剪
        depth_clipped = np.clip(depth_resized, display_vmin, display_vmax)
        depth_masked = np.where(mask_resized, depth_clipped, np.nan)
        
        path_pure = out_dir / f"{stem}_pure.png"
        path_probes = out_dir / f"{stem}_probes.png"

        # --- A. 纯净版 (无白边，无标尺) ---
        # 强制设置 figure 大小为像素比例
        fig_clean, ax_clean = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
        fig_clean.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax_clean.imshow(depth_masked, cmap=self.cmap, vmin=display_vmin, vmax=display_vmax)
        ax_clean.axis('off')
        fig_clean.savefig(path_pure, pad_inches=0)
        plt.close(fig_clean)

        # --- B. 探针版 (无标尺，仅深度值) ---
        fig, ax = plt.subplots(figsize=(TARGET_SIZE[0]/100, TARGET_SIZE[1]/100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(depth_masked, cmap=self.cmap, vmin=display_vmin, vmax=display_vmax)
        ax.axis('off')

        # 5. 绘制探针 (需要重新计算探针在缩放后的坐标)
        orig_h, orig_w = depth_array.shape
        scale_x = TARGET_SIZE[0] / orig_w
        scale_y = TARGET_SIZE[1] / orig_h

        colors = ['#FF3366', '#00E676'] 
        for pt_idx, (px, py) in enumerate(target_probes):
            # 映射坐标到新尺寸
            new_x, new_y = int(px * scale_x), int(py * scale_y)
            
            c = colors[pt_idx % len(colors)]
            ax.plot(new_x, new_y, marker='o', markerfacecolor=c, markeredgecolor='white', 
                    markersize=12, markeredgewidth=2, zorder=5)
            
            # 获取缩放后对应位置的原始深度值
            val = depth_resized[new_y, new_x]
            text_str = f"{val:.1f}m"
            
            # 悬浮文字
            y_offset = -40 if pt_idx == 0 else 40 
            ax.text(new_x, new_y + y_offset, text_str, color='black', fontsize=14, fontweight='bold',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor="none", alpha=0.8))

        fig.savefig(path_probes, pad_inches=0)
        plt.close(fig)

# ================= 3. 数据加载与处理 =================

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

def resolve_depth_paths(targets):
    resolved = []
    if isinstance(targets, (str, Path)):
        targets = [targets]
    for t in targets:
        p = Path(t).resolve()
        if p.is_file() and p.suffix.lower() in ['.npy', '.png']:
            resolved.append(p)
    return list(dict.fromkeys(resolved))



if __name__ == "__main__":
    print(f"{'='*60}\n📊 深度图严格对齐渲染器 (Support Duplicates)\n{'='*60}")
    
    # 1. 解析路径（允许重复）
    target_paths = resolve_depth_paths(INPUT_TARGETS)
    if not target_paths:
        print("❌ 未找到任何文件。")
        sys.exit(1)
        
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载数据
    loaded_datasets = []
    all_valid_depths = [] 
    img_shape = None
    global_mask = None

    for file_path in target_paths:
        data = load_depth_data(file_path)
        if data is None: continue
        
        loaded_datasets.append((file_path, data))
        mask = get_valid_mask(data)
        valid_pixels = data[mask]
        if len(valid_pixels) > 0:
            all_valid_depths.append(valid_pixels)
        
        if img_shape is None:
            img_shape = data.shape
            global_mask = mask
        elif data.shape == img_shape:
            global_mask &= mask

    # 3. 计算全局范围 (vmin/vmax)
    if GLOBAL_VMIN is not None and GLOBAL_VMAX is not None:
        final_vmin, final_vmax = GLOBAL_VMIN, GLOBAL_VMAX
    elif len(all_valid_depths) > 0:
        concatenated_depths = np.concatenate(all_valid_depths)
        final_vmin = np.percentile(concatenated_depths, 1)
        final_vmax = np.percentile(concatenated_depths, 99)
    else:
        final_vmin, final_vmax = 0.1, 100.0

    # 4. 确定探针
    h, w = img_shape if img_shape else (1080, 1920)
    cx, cy = w // 2, h // 2
    offset_x = w // 4
    pt1 = get_nearest_valid_point(cx - offset_x, cy, global_mask) if global_mask is not None else (cx - offset_x, cy)
    pt2 = get_nearest_valid_point(cx + offset_x, cy, global_mask) if global_mask is not None else (cx + offset_x, cy)
    
    
    if USE_MANUAL_PROBES and MANUAL_PROBES:
        print(f"📍 正在从手动坐标进行稀疏吸附采样...")
        global_probes = []
        h_orig, w_orig = img_shape if img_shape else (960, 1280)
        
        for px, py in MANUAL_PROBES:
            # 1. 越界检查
            px = max(0, min(px, w_orig - 1))
            py = max(0, min(py, h_orig - 1))
            
            # 2. 吸附：寻找最近的有效深度点
            # 这一步非常关键，否则在稀疏图中你大概率会点中“无值”区域
            pt = get_nearest_valid_point(px, py, global_mask)
            if pt is not None:
                global_probes.append(pt)
                print(f"✅ 成功锁定 {len(global_probes)} 个有效深度探针。")
    else:
        print("🤖 正在自动计算最佳探针位置...")
        h, w = img_shape if img_shape else (1080, 1920)
        cx, cy = w // 2, h // 2
        offset_x = w // 4
        pt1 = get_nearest_valid_point(cx - offset_x, cy, global_mask) if global_mask is not None else (cx - offset_x, cy)
        pt2 = get_nearest_valid_point(cx + offset_x, cy, global_mask) if global_mask is not None else (cx + offset_x, cy)
        global_probes = [pt for pt in [pt1, pt2] if pt is not None]
        # global_probes = [pt for pt in [pt1, pt2] if pt is not None]

    # 5. 渲染循环（加入 idx 适配重复名称）
    renderer = DepthRenderer()
    for idx, (file_path, data) in enumerate(tqdm(loaded_datasets, desc="Rendering")):
        # 唯一标识符：序号_父目录_文件名
        unique_stem = f"{idx:02d}_{file_path.parent.name}_{file_path.stem}"
        
        renderer.render_trio(
            depth_array=data,
            stem=unique_stem,
            out_dir=out_dir,
            target_probes=global_probes,
            vmin=final_vmin,
            vmax=final_vmax
        )
            
    print(f"\n✅ 渲染完成！输出目录: {out_dir}")
