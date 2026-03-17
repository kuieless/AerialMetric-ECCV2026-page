import numpy as np
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================================
# 配置区域
# =================================================================================
USE_MEDIAN_SCALING = False
BATCH_SIZE = 32
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 800  # 根据你的代码调整了最大深度

# --- 直方图统计配置 ---
HIST_BINS = 400              # 将深度范围分割成多少个柱子
HIST_RANGE = (0, 800)        # 统计的深度范围 (米)
OUTPUT_PLOT_DIR = "./histogram_plots_scenes" # 输出目录

# 数据集路径配置 (直接复用你的列表)
EVALUATION_PATHS = [
    {
        'name': 'hav',
        'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize101500-no300/hav/depth_npy/hav',
        'gt_dir': '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_hav'
    },
    {
        'name': 'smbu',
        'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize101500-no300/SMBU/depth_npy/SMBU',
        'gt_dir': '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_SMBU'
    },
    {
        'name': 'sztu',
        'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize101500-no300/sztu/depth_npy/sztu',
        'gt_dir': '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_sztu'
    },
    {
        'name': 'lfls2',
        'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize101500-no300/lfls2/depth_npy/lfls2',
        'gt_dir': '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_lfls2'
    },
    # {
    #     'name': 'upper',
    #     'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize101500-no300/upper/depth_npy/',
    #     'gt_dir': '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_upper'
    # },   
]

# =================================================================================
# 1. Dataset 定义 (保持简单高效)
# =================================================================================

class PairedNpyDataset(Dataset):
    def __init__(self, pred_dir, gt_dir):
        all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        self.pred_files = []
        self.gt_files = []
        for pred_path in all_pred_files:
            basename = os.path.splitext(os.path.basename(pred_path))[0]
            gt_filename = f"{basename}.npy"
            gt_path = os.path.join(gt_dir, gt_filename)
            if os.path.exists(gt_path):
                self.pred_files.append(pred_path)
                self.gt_files.append(gt_path)
                
        if len(self.pred_files) == 0:
            print(f"警告: 在 {pred_dir} 未找到匹配文件对。")

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        pred_np = np.load(self.pred_files[idx]).astype(np.float32)
        gt_np = np.load(self.gt_files[idx]).astype(np.float32)

        if pred_np.ndim == 3: pred_np = np.squeeze(pred_np)
        if gt_np.ndim == 3: gt_np = np.squeeze(gt_np)
            
        pred_tensor = torch.from_numpy(pred_np)
        gt_tensor = torch.from_numpy(gt_np)
        
        return pred_tensor, gt_tensor

# =================================================================================
# 2. 核心统计逻辑
# =================================================================================

def update_histogram_stats(gt_batch, pred_batch, mask_batch, scene_name, accumulators):
    """
    计算当前 Batch 的直方图，并累加到对应场景的统计器中。
    """
    # 转为 numpy 处理，展平所有像素
    valid_mask_np = mask_batch.cpu().numpy()
    
    # 如果整个 batch 都没有有效像素，跳过
    if valid_mask_np.sum() == 0:
        return

    gt_np = gt_batch.cpu().numpy()[valid_mask_np]
    pred_np = pred_batch.cpu().numpy()[valid_mask_np]
    
    # 计算直方图 (不存储像素，只存储频数)
    hist_gt, _ = np.histogram(gt_np, bins=HIST_BINS, range=HIST_RANGE)
    hist_pred, _ = np.histogram(pred_np, bins=HIST_BINS, range=HIST_RANGE)
    
    # 1. 累加到当前场景 (Scene)
    if scene_name not in accumulators:
        accumulators[scene_name] = {
            'gt_counts': np.zeros(HIST_BINS, dtype=np.float64),
            'pred_counts': np.zeros(HIST_BINS, dtype=np.float64)
        }
    accumulators[scene_name]['gt_counts'] += hist_gt
    accumulators[scene_name]['pred_counts'] += hist_pred
    
    # 2. 累加到全局 (ALL)
    if 'ALL' not in accumulators:
        accumulators['ALL'] = {
            'gt_counts': np.zeros(HIST_BINS, dtype=np.float64),
            'pred_counts': np.zeros(HIST_BINS, dtype=np.float64)
        }
    accumulators['ALL']['gt_counts'] += hist_gt
    accumulators['ALL']['pred_counts'] += hist_pred

# =================================================================================
# 3. 绘图函数
# =================================================================================

def plot_and_save(accumulators, bin_edges, output_dir):
    """
    根据累积的统计数据绘制直方图
    """
    sns.set_theme(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算 bin 的中心点用于绘图 (x轴)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # 获取所有场景名，确保 'ALL' 在最后
    scene_keys = sorted([k for k in accumulators.keys() if k != 'ALL'])
    if 'ALL' in accumulators:
        scene_keys.append('ALL')
        
    print(f"\n开始绘图，将保存至: {output_dir}")
    
    for scene_name in scene_keys:
        stats = accumulators[scene_name]
        gt_counts = stats['gt_counts']
        pred_counts = stats['pred_counts']
        
        # 归一化 (Probability Density)
        gt_sum = gt_counts.sum()
        pred_sum = pred_counts.sum()
        
        if gt_sum == 0: continue

        gt_density = gt_counts / gt_sum
        pred_density = pred_counts / pred_sum
        
        # --- 绘图 1: 线性坐标 (Linear Scale) ---
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(bin_centers, gt_density, label='Ground Truth', color='black', linewidth=2, alpha=0.8)
        plt.plot(bin_centers, pred_density, label='Prediction', color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.fill_between(bin_centers, gt_density, alpha=0.3, color='grey')
        plt.fill_between(bin_centers, pred_density, alpha=0.2, color='red')
        
        title_str = f"Depth Distribution - Scene: {scene_name}"
        plt.title(title_str, fontsize=16)
        plt.xlabel("Depth (meters)", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        plt.legend(fontsize=12)
        plt.xlim(HIST_RANGE[0], HIST_RANGE[1])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        save_name = f"hist_scene_{scene_name}_linear.png"
        plt.savefig(os.path.join(output_dir, save_name), bbox_inches='tight')
        plt.close()
        
        # --- 绘图 2: 对数坐标 (Log Scale) ---
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(bin_centers, gt_density, label='Ground Truth', color='black', linewidth=2)
        plt.plot(bin_centers, pred_density, label='Prediction', color='red', linewidth=2, linestyle='--')
        
        plt.title(f"{title_str} (Log Scale)", fontsize=16)
        plt.xlabel("Depth (meters)", fontsize=12)
        plt.ylabel("Log Density", fontsize=12)
        plt.yscale('log') # 开启 Log
        plt.legend(fontsize=12)
        plt.xlim(HIST_RANGE[0], HIST_RANGE[1])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        save_name = f"hist_scene_{scene_name}_log.png"
        plt.savefig(os.path.join(output_dir, save_name), bbox_inches='tight')
        plt.close()
        
        print(f" -> 已保存场景 {scene_name} 的图表")

# =================================================================================
# 4. 主循环
# =================================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA GPU。")
        exit()
    
    device = torch.device("cuda")
    
    # 初始化累加器字典
    # 结构: { 'hav': {'gt': ..., 'pred': ...}, 'ALL': ... }
    accumulators = {} 
    
    # 生成 Bin 边缘
    bin_edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)

    print(f"开始统计场景深度分布...")
    print(f"范围: {HIST_RANGE} m, Bins数量: {HIST_BINS}")
    print(f"可视化目录: {OUTPUT_PLOT_DIR}")
    
    for path_info in tqdm(EVALUATION_PATHS, desc="总进度", unit="scene"):
        scene_name = path_info['name']
        pred_dir = path_info['pred_dir']
        gt_dir = path_info['gt_dir']

        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            print(f"跳过无效路径: {scene_name}")
            continue

        try:
            dataset = PairedNpyDataset(pred_dir, gt_dir)
            # 使用 batch_size 加速 I/O
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                    num_workers=NUM_WORKERS, pin_memory=True)
        except Exception as e:
            print(f"初始化场景 {scene_name} 失败: {e}")
            continue

        pbar_desc = f"处理 {scene_name}"
        for pred_batch, gt_batch in tqdm(dataloader, desc=pbar_desc, leave=False):
            
            # 数据上 GPU
            pred_batch, gt_batch = pred_batch.to(device), gt_batch.to(device)
            
            # GPU 上插值 (如果尺寸不匹配)
            target_shape = gt_batch.shape[-2:]
            if pred_batch.shape[-2:] != target_shape:
                pred_batch = pred_batch.unsqueeze(1) 
                pred_batch = F.interpolate(pred_batch, size=target_shape, mode='bilinear', align_corners=False)
                pred_batch = pred_batch.squeeze(1)

            # 掩码处理
            mask_batch = (gt_batch > MIN_EVAL_DEPTH) & (gt_batch < MAX_EVAL_DEPTH)
            
            # 中位数定标 (可选)
            if USE_MEDIAN_SCALING:
                gt_nan = gt_batch.clone()
                pred_nan = pred_batch.clone()
                gt_nan[~mask_batch] = torch.nan
                pred_nan[~mask_batch] = torch.nan
                median_gt = torch.nanmedian(gt_nan, dim=[1, 2]).values 
                median_pred = torch.nanmedian(pred_nan, dim=[1, 2]).values 
                ratio = median_gt / median_pred
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
                pred_batch = pred_batch * ratio.view(-1, 1, 1)
            
            # 裁剪
            pred_batch = torch.clamp(pred_batch, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
            
            # ---> 核心：更新统计 (按场景名) <---
            update_histogram_stats(gt_batch, pred_batch, mask_batch, scene_name, accumulators)

    # 绘图
    if accumulators:
        plot_and_save(accumulators, bin_edges, OUTPUT_PLOT_DIR)
        print("\n所有场景直方图绘制完成！")
    else:
        print("\n未收集到有效数据，无法绘图。")