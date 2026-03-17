import numpy as np
import cv2
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime



PATHS_MOGE_LARGE = [




]


TASKS = [
    #     {
    #     "task_name": "Moge2-Base-Absolutedepthrange",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/Moge/Infer3-b.txt",
    #     "use_median_scaling": False,  # 关闭定标
    #     "paths": PATHS_MOGE_BASE     # 使用上面定义的第一组列表
    # },

    # # --- 任务 B: Moge2 Large (有定标 / 相对尺度) ---
    # {
    #     "task_name": "Moge2-Base-MedianScalingdepthrange",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/Moge/Infer3-b-MS.txt",
    #     "use_median_scaling": True,   # 开启定标
    #     "paths": PATHS_MOGE_BASE     # 复用同一组数据，只是参数不同
    # },

            {
        "task_name": "Moge2-large-Absolutedepthrange",
        "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/Moge/Infer-AAA-down-l-finecheck-1k.txt",
        "use_median_scaling": False,  # 关闭定标
        "paths": PATHS_MOGE_LARGE     # 使用上面定义的第一组列表
    },

    # --- 任务 B: Moge2 Large (有定标 / 相对尺度) ---
    {
        "task_name": "Moge2-large-MedianScalingdepthrange",
        "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/Moge/Infer-AAA-down-l-finecheck-1k-MS.txt",
        "use_median_scaling": True,   # 开启定标
        "paths": PATHS_MOGE_LARGE     # 复用同一组数据，只是参数不同
    },

    #         {
    #     "task_name": "Moge2-SMALL-Absolutedepthrange",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/Moge/Infer3-s.txt",
    #     "use_median_scaling": False,  # 关闭定标
    #     "paths": PATHS_MOGE_SMALL     # 使用上面定义的第一组列表
    # },

    # # --- 任务 B: Moge2 Large (有定标 / 相对尺度) ---
    # {
    #     "task_name": "Moge2-SMALL-MedianScalingdepthrange",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/Moge/Infer3-s-MS.txt",
    #     "use_median_scaling": True,   # 开启定标
    #     "paths": PATHS_MOGE_SMALL     # 复用同一组数据，只是参数不同
    # },
]
# -------------------------------------------------------------------------
# 新增: 定义深度分段区间 (左闭右开: [min, max))
# -------------------------------------------------------------------------
DEPTH_BINS = [
    (0, 50),
    (50, 120),
    (120, 250),
    (250, 400),
    (400, 500)
]
# 用于报告显示的标签
BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]

# =================================================================================
# 核心代码
# =================================================================================
BATCH_SIZE = 32
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 500

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
            raise ValueError(f"No matched files in {pred_dir} and {gt_dir}")

    def __len__(self): return len(self.pred_files)
    
    def __getitem__(self, idx):
        pred = torch.from_numpy(np.load(self.pred_files[idx]).astype(np.float32)).squeeze()
        gt = torch.from_numpy(np.load(self.gt_files[idx]).astype(np.float32)).squeeze()
        return pred, gt

def compute_errors_torch_bins(gt, pred, valid_mask):
    """
    修改版: 同时计算所有分段的指标
    返回形状: [Batch, Num_Bins + 1, 8] 
    其中 Num_Bins + 1 的最后一个索引存放 'Overall' (全深度) 的结果
    """
    # 1. 预处理数据 (Clamping)
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
    # 2. 预计算所有像素的 Error Maps (避免在循环中重复计算，提高速度)
    # RMSE Map
    rmse_map = (gt - pred) ** 2
    # RMSE Log Map
    rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
    # Abs Rel Map
    abs_rel_map = torch.abs(gt - pred) / gt_c
    # Sq Rel Map
    sq_rel_map = ((gt - pred) ** 2) / gt_c
    # Threshold Ratio
    thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
    # 定义所有要统计的区间：用户定义的 Bins + 全局 Overall
    # 这里的 Overall 就是你原代码的逻辑，范围是 valid_mask 定义的范围
    all_ranges = DEPTH_BINS + [("Overall", "Overall")] 
    
    batch_results = []
    
    for (b_min, b_max) in all_ranges:
        # 3. 生成当前 Bin 的 Mask
        if b_min == "Overall":
            # Overall 仅使用传入的 valid_mask (已经包含 min/max truncate)
            current_mask = valid_mask
        else:
            # 特定 Bin: 基础 Mask AND 深度区间
            # 注意: gt >= b_min AND gt < b_max
            current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
            
        # 4. 基于当前 Mask 进行聚合 (Sum)
        # 注意：如果不 mask 掉，sum 会把无效像素也加进去，所以必须用 mask
        # 技巧：用 where 或者 mask 乘法。这里为了简单和显存，直接利用 mask 索引求和并不方便(因为Batch维还在)
        # 我们使用乘法 mask (float)
        mask_f = current_mask.float()
        valid_pixel_count = mask_f.sum(dim=[1, 2])
        
        # 计算各项 Sum
        # a1, a2, a3
        a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
        a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
        a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
        # Errors
        rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
        rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
        abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
        sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])
        
        # Stack 得到 [Batch, 8]
        bin_res = torch.stack([abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count], dim=1)
        batch_results.append(bin_res)
        
    # Stack 所有 Bins -> [Batch, Num_Bins+1, 8]
    return torch.stack(batch_results, dim=1)

def compute_metrics_from_sums(sums):
    total_valid = sums[7]
    if total_valid <= 0: 
        return np.zeros(7) # 如果该区间没有有效像素，返回全0
    return np.array([
        sums[0]/total_valid, sums[1]/total_valid, np.sqrt(sums[2]/total_valid),
        sums[3]/total_valid, sums[4]/total_valid, sums[5]/total_valid, sums[6]/total_valid
    ])

def evaluate_single_scene(pred_dir, gt_dir, scene_name, use_ms):
    device = torch.device("cuda")
    try:
        ds = PairedNpyDataset(pred_dir, gt_dir)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    except ValueError:
        print(f"  [Skipped] Empty/Mismatch: {scene_name}"); return None

    all_errs = []
    pbar_desc = f"  -> {scene_name}"
    
    # 这里的 all_errs 将会存储 [Batch, Num_Bins+1, 8] 形状的数据
    for pred_b, gt_b in tqdm(dl, desc=pbar_desc, leave=False):
        pred_b, gt_b = pred_b.to(device), gt_b.to(device)
        
        if pred_b.shape[-2:] != gt_b.shape[-2:]:
            pred_b = F.interpolate(pred_b.unsqueeze(1), size=gt_b.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        
        mask = (gt_b > MIN_EVAL_DEPTH) & (gt_b < MAX_EVAL_DEPTH)
        
        if use_ms:
            B = gt_b.shape[0]
            gt_flat = gt_b.clone(); gt_flat[~mask] = float('nan'); gt_flat = gt_flat.view(B, -1)
            pred_flat = pred_b.clone(); pred_flat[~mask] = float('nan'); pred_flat = pred_flat.view(B, -1)
            med_gt = torch.nanmedian(gt_flat, dim=1).values
            med_pred = torch.nanmedian(pred_flat, dim=1).values
            ratio = torch.nan_to_num(med_gt / med_pred, nan=1.0, posinf=1.0, neginf=1.0)
            pred_b = pred_b * ratio.view(B, 1, 1)

        pred_b.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
        
        # 调用新的 Bin 计算函数
        # 结果形状: [Batch, Num_Bins+1, 8] -> 转为 numpy
        batch_res = compute_errors_torch_bins(gt_b, pred_b, mask).cpu().numpy()
        all_errs.append(batch_res)
            
    if not all_errs: return None
    # 拼接所有 batch: [Total_Images, Num_Bins+1, 8]
    return np.concatenate(all_errs, axis=0)

def format_line(name, m, indent=0):
    sp = " " * indent
    # 为了防止NaN (当某区间像素数为0时)，做个简单处理
    m = np.nan_to_num(m) 
    return "{:<50} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} |".format(
        sp + name[-50+indent:], m[0], m[1], m[2], m[3], m[4], m[5], m[6])

# =================================================================================
# 主循环
# =================================================================================
if __name__ == '__main__':
    if not torch.cuda.is_available(): exit("No CUDA device.")
    
    # 所有的标签：Bins + Overall
    ALL_LABELS = BIN_LABELS + ["Overall"]
    
    print(f"Batch Inference Started. Total Tasks: {len(TASKS)}")
    
    for i, task in enumerate(TASKS):
        t_name = task['task_name']
        t_out = task['output_file']
        t_ms = task['use_median_scaling']
        t_paths = task['paths']
        
        print(f"\n[{i+1}/{len(TASKS)}] Processing Task: {t_name}")
        print(f"  Mode: {'Median Scaling' if t_ms else 'Absolute'}")
        
        os.makedirs(os.path.dirname(t_out), exist_ok=True)
        
        results_map = {}
        # 这里的 all_errs_list 将存储每个场景的 [N, Bins, 8] 数据
        all_errs_list = []
        
        for path_dict in t_paths:
            s_name = path_dict['name']
            p_dir = path_dict['pred_dir']
            g_dir = path_dict['gt_dir']

            if not os.path.isdir(p_dir) or not os.path.isdir(g_dir):
                print(f"  [Error] Path not found for scene: {s_name}")
                continue

            # 计算该场景所有图片的误差
            scene_errs = evaluate_single_scene(p_dir, g_dir, s_name, t_ms)
            
            if scene_errs is not None:
                # scene_errs shape: [N_imgs, Num_Bins+1, 8]
                # 我们先存起来，后面统一算 sum
                results_map[s_name] = scene_errs
                all_errs_list.append(scene_errs)
        
        # --- 生成详细报告 ---
        lines = []
        header = "{:<50} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |".format("Scene / Depth Bin", "AbsRel", "SqRel", "RMSE", "RMSElog", "a1", "a2", "a3")
        sep = "-" * len(header)
        lines += [f"Task: {t_name}", f"Date: {datetime.now()}", f"Median Scaling: {t_ms}", "="*100, header, sep]
        
        # 1. 遍历每个场景
        for s_name in sorted(results_map.keys()):
            # s_data: [N_imgs, Num_Bins+1, 8]
            s_data = results_map[s_name] 
            
            # 对该场景的所有图片求和 -> [Num_Bins+1, 8]
            s_sums = s_data.sum(axis=0) 
            
            # 打印场景标题行 (显示 Overall)
            # 最后一个索引是 Overall
            overall_mean = compute_metrics_from_sums(s_sums[-1]) 
            lines.append(format_line(f"> {s_name} (All)", overall_mean))
            
            # 打印该场景下的分段详情
            for bin_idx, bin_label in enumerate(BIN_LABELS):
                bin_mean = compute_metrics_from_sums(s_sums[bin_idx])
                # 缩进显示，区分层级
                lines.append(format_line(f"   [{bin_label}]", bin_mean, indent=3))
            
            lines.append(sep) # 场景间的分隔线
            
        # 2. 计算整个 Dataset 的平均 (Total Average)
        if all_errs_list:
            # 拼接所有场景 -> [Total_All_Imgs, Num_Bins+1, 8]
            total_data = np.concatenate(all_errs_list, axis=0)
            total_sums = total_data.sum(axis=0) # -> [Num_Bins+1, 8]
            
            lines.append("="*100)
            lines.append(">>> DATASET TOTAL AVERAGE <<<")
            lines.append(sep)
            
            # 打印 Total Overall
            total_overall_mean = compute_metrics_from_sums(total_sums[-1])
            lines.append(format_line("TOTAL (All Depths)", total_overall_mean))
            
            # 打印 Total Bins
            for bin_idx, bin_label in enumerate(BIN_LABELS):
                bin_mean = compute_metrics_from_sums(total_sums[bin_idx])
                lines.append(format_line(f"TOTAL [{bin_label}]", bin_mean))
            
            lines.append("="*100)
            
        with open(t_out, 'w') as f: f.write("\n".join(lines))
        print(f"  -> Report saved to {t_out}")
        
        del results_map, all_errs_list
        torch.cuda.empty_cache()
        
    print("\nAll tasks completed.")