import numpy as np
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================================
# 配置区域 (Configuration)
# =================================================================================
# 核心配置 (与评估脚本保持一致)
USE_MEDIAN_SCALING = False
BATCH_SIZE = 32
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 400  # 关注的深度范围上限

# --- 直方图统计配置 ---
HIST_BINS = 200              # 将 0-MAX_EVAL_DEPTH 分割成多少个柱子
HIST_RANGE = (0, 400)        # 统计的深度范围 (米)
OUTPUT_PLOT_DIR = "./histogram_plots_v1" # 直方图保存目录

# 定义需要统计的角度
PITCH_ANGLES = [0, 15, 30, 45, 60]

# 数据集路径 (直接复用你的配置)
EVALUATION_PATHS = [
    # {
    #     'name': '11scene',
    #     'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize10400-666-101500-no300/images_downsampled/depth_npy/images_downsampled',
    #     'gt_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-down/npy'
    # },

           {
            'name': '11scene',
            'pred_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-Moge-out',
            'gt_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-down/npy'
        },
]

# =================================================================================
# 1. 数据加载类 (完全复用原脚本，确保逻辑一致)
# =================================================================================

FILENAME_PATTERN = re.compile(r'(.+?)-(\d+)-(\d{6})$') 

class PairedNpyDataset(Dataset):
    def __init__(self, pred_dir, gt_dir, scene_base_name):
        all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        self.pred_files = []
        self.gt_files = []
        self.metadata = [] 
        self.scene_base_name = scene_base_name

        print(f"正在扫描: {pred_dir} ...")
        for pred_path in all_pred_files:
            basename = os.path.splitext(os.path.basename(pred_path))[0]
            gt_filename = f"{basename}.npy"
            gt_path = os.path.join(gt_dir, gt_filename)
            
            match = FILENAME_PATTERN.match(basename)
            if not match:
                continue

            scene_name_part = match.group(1) 
            angle = int(match.group(2))

            if os.path.exists(gt_path):
                self.pred_files.append(pred_path)
                self.gt_files.append(gt_path)
                self.metadata.append({
                    'full_scene': self.scene_base_name,
                    'sub_scene': scene_name_part,
                    'angle': angle
                })
        
        if len(self.pred_files) == 0:
            print(f"警告: 在 {pred_dir} 未找到匹配文件。")
        else:
            print(f" -> 找到 {len(self.pred_files)} 个有效的 PRED/GT 文件对。")

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        pred_np = np.load(self.pred_files[idx]).astype(np.float32)
        gt_np = np.load(self.gt_files[idx]).astype(np.float32)

        if pred_np.ndim == 3: pred_np = np.squeeze(pred_np)
        if gt_np.ndim == 3: gt_np = np.squeeze(gt_np)
            
        pred_tensor = torch.from_numpy(pred_np)
        gt_tensor = torch.from_numpy(gt_np)
        
        meta = self.metadata[idx].copy() 
        return pred_tensor, gt_tensor, meta

def collate_fn_with_meta(batch):
    pred_tensors = [item[0] for item in batch]
    gt_tensors = [item[1] for item in batch]
    metadata = [item[2] for item in batch]
    return pred_tensors, gt_tensors, metadata

# =================================================================================
# 2. 核心统计逻辑: 在线直方图累积
# =================================================================================

def update_histogram_stats(gt_batch, pred_batch, mask_batch, meta_batch, accumulators):
    """
    计算当前 Batch 的直方图，并累加到全局统计器 accumulators 中。
    """
    # 转为 numpy 处理
    gt_np = gt_batch.cpu().numpy()
    pred_np = pred_batch.cpu().numpy()
    mask_np = mask_batch.cpu().numpy()
    
    # 遍历 Batch 中的每一张图 (因为每张图可能有不同的角度)
    for i in range(len(meta_batch)):
        angle = meta_batch[i]['angle']
        
        # 提取当前图的有效像素
        valid_mask = mask_np[i]
        if valid_mask.sum() == 0:
            continue
            
        curr_gt = gt_np[i][valid_mask]
        curr_pred = pred_np[i][valid_mask]
        
        # 计算直方图 (不存储像素，只存储频数)
        # density=False, 此时返回的是计数，我们最后再归一化
        hist_gt, _ = np.histogram(curr_gt, bins=HIST_BINS, range=HIST_RANGE)
        hist_pred, _ = np.histogram(curr_pred, bins=HIST_BINS, range=HIST_RANGE)
        
        # 初始化该角度的累加器
        if angle not in accumulators:
            accumulators[angle] = {
                'gt_counts': np.zeros(HIST_BINS, dtype=np.float64),
                'pred_counts': np.zeros(HIST_BINS, dtype=np.float64)
            }
        
        # 累加
        accumulators[angle]['gt_counts'] += hist_gt
        accumulators[angle]['pred_counts'] += hist_pred
        
        # 同时维护一个 'ALL' 角度的总和
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
    
    # 获取所有待绘制的 Key (按角度排序, ALL 放最后)
    keys = sorted([k for k in accumulators.keys() if isinstance(k, int)])
    if 'ALL' in accumulators:
        keys.append('ALL')
        
    print(f"\n开始绘图，将保存至: {output_dir}")
    
    for angle in keys:
        stats = accumulators[angle]
        gt_counts = stats['gt_counts']
        pred_counts = stats['pred_counts']
        
        # 归一化 (转换为概率密度 Probability Density)
        gt_sum = gt_counts.sum()
        pred_sum = pred_counts.sum()
        
        if gt_sum == 0: continue

        gt_density = gt_counts / gt_sum
        pred_density = pred_counts / pred_sum
        
        # --- 绘图 1: 线性坐标 ---
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, gt_density, label='Ground Truth', color='black', linewidth=2, alpha=0.8)
        plt.plot(bin_centers, pred_density, label='Prediction', color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        # 填充颜色增加可视性
        plt.fill_between(bin_centers, gt_density, alpha=0.3, color='grey')
        plt.fill_between(bin_centers, pred_density, alpha=0.2, color='red')
        
        title_str = f"Depth Distribution - Angle {angle}°" if angle != 'ALL' else "Depth Distribution - All Angles"
        plt.title(title_str, fontsize=15)
        plt.xlabel("Depth (meters)", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        plt.legend(fontsize=12)
        plt.xlim(HIST_RANGE[0], HIST_RANGE[1])
        
        save_name = f"hist_angle_{angle}_linear.png"
        plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- 绘图 2: 对数坐标 (Y轴 Log) ---
        # 这有助于查看长尾分布或稀疏区域
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, gt_density, label='Ground Truth', color='black', linewidth=2)
        plt.plot(bin_centers, pred_density, label='Prediction', color='red', linewidth=2, linestyle='--')
        
        plt.title(f"{title_str} (Log Scale)", fontsize=15)
        plt.xlabel("Depth (meters)", fontsize=12)
        plt.ylabel("Log Density", fontsize=12)
        plt.yscale('log') # <--- 关键
        plt.legend(fontsize=12)
        plt.xlim(HIST_RANGE[0], HIST_RANGE[1])
        
        save_name = f"hist_angle_{angle}_log.png"
        plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" -> 已保存角度 {angle} 的图表")

# =================================================================================
# 4. 主循环
# =================================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA GPU。")
        exit()
    
    device = torch.device("cuda")
    
    # 初始化累加器字典
    # 结构: { 0: {'gt': [], 'pred': []}, 15: ... }
    accumulators = {} 
    
    # 生成 Bin 边缘 (只生成一次)
    bin_edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)

    print(f"开始统计直方图分布...")
    print(f"范围: {HIST_RANGE} m, Bins数量: {HIST_BINS}")
    
    for path_info in tqdm(EVALUATION_PATHS, desc="总进度", unit="scene"):
        scene_base_name = path_info['name']
        pred_dir = path_info['pred_dir']
        gt_dir = path_info['gt_dir']

        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            continue

        # 初始化 Dataset 和 Loader
        try:
            dataset = PairedNpyDataset(pred_dir, gt_dir, scene_base_name)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                    num_workers=NUM_WORKERS, pin_memory=True, 
                                    collate_fn=collate_fn_with_meta)
        except ValueError:
            continue

        # 遍历数据
        for pred_list, gt_list, metadata_list in tqdm(dataloader, desc=f"读取 {scene_base_name}", leave=False):
            
            # 堆叠 Batch
            pred_batch = torch.stack(pred_list).to(device) # (B, H, W)
            gt_batch = torch.stack(gt_list).to(device)     # (B, H, W)
            
            # 插值 (如果尺寸不匹配)
            if pred_batch.shape[-2:] != gt_batch.shape[-2:]:
                pred_batch = F.interpolate(pred_batch.unsqueeze(1), size=gt_batch.shape[-2:], 
                                           mode='bilinear', align_corners=False).squeeze(1)

            # 掩码处理
            mask_batch = (gt_batch > MIN_EVAL_DEPTH) & (gt_batch < MAX_EVAL_DEPTH)
            
            # Median Scaling (如果需要保持和评估一致)
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
            
            # ---> 核心：更新统计 <---
            update_histogram_stats(gt_batch, pred_batch, mask_batch, metadata_list, accumulators)

    # 所有数据跑完后，画图
    if accumulators:
        plot_and_save(accumulators, bin_edges, OUTPUT_PLOT_DIR)
        print("\n所有直方图绘制完成！")
    else:
        print("\n未收集到有效数据，无法绘图。")