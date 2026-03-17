import numpy as np
import cv2
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

# =================================================================================
# 配置区域：在这里定义您的数据列表和任务
# =================================================================================

# -------------------------------------------------------------------------
# 1. 定义数据路径列表 (您可以定义多组，每组包含不同的场景)
# -------------------------------------------------------------------------

# 第一组路径：例如 Moge2 Large 的输出
PATHS_MOGE_LARGE = [
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer2-l-normal-out/SYS/',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer2-l-normal-out/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
    },

    # ... 您可以在这里继续粘贴更多字典 ...
]

# 第二组路径：例如 Metric3D 的输出 (如果路径不一样的话)
# 如果路径完全一样，只是测不同的参数，可以复用上面的 list，不用重写
PATHS_MOGE_BASE = [
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer2-b-normal-out/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer2-b-normal-out/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
    },
    # ...

    
]
PATHS_MOGE_SMALL = [
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer2-s-normal-out/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer2-s-normal-out/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
    },
    # ...

    
]

# -------------------------------------------------------------------------
# 2. 定义任务列表 (TASKS)
# -------------------------------------------------------------------------
TASKS = [
    # --- 任务 A: Moge2 Large (无定标 / 绝对尺度) ---
    {
        "task_name": "Moge2-Large-Absolute",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer-l.txt",
        "use_median_scaling": False,  # 关闭定标
        "paths": PATHS_MOGE_LARGE     # 使用上面定义的第一组列表
    },

    # --- 任务 B: Moge2 Large (有定标 / 相对尺度) ---
    {
        "task_name": "Moge2-Large-MedianScaling",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer-l-MS.txt",
        "use_median_scaling": True,   # 开启定标
        "paths": PATHS_MOGE_LARGE     # 复用同一组数据，只是参数不同
    },


        {
        "task_name": "Moge2-Base-Absolute",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer-b.txt",
        "use_median_scaling": False,  # 关闭定标
        "paths": PATHS_MOGE_BASE     # 使用上面定义的第一组列表
    },

    # --- 任务 B: Moge2 Large (有定标 / 相对尺度) ---
    {
        "task_name": "Moge2-Base-MedianScaling",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer-b-MS.txt",
        "use_median_scaling": True,   # 开启定标
        "paths": PATHS_MOGE_BASE     # 复用同一组数据，只是参数不同
    },


            {
        "task_name": "Moge2-SMALL-Absolute",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer-s.txt",
        "use_median_scaling": False,  # 关闭定标
        "paths": PATHS_MOGE_SMALL     # 使用上面定义的第一组列表
    },

    # --- 任务 B: Moge2 Large (有定标 / 相对尺度) ---
    {
        "task_name": "Moge2-SMALL-MedianScaling",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/infer2/Infer-s-MS.txt",
        "use_median_scaling": True,   # 开启定标
        "paths": PATHS_MOGE_SMALL     # 复用同一组数据，只是参数不同
    },


]

# =================================================================================
# 核心代码 (无需修改)
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
        # 纯IO，不进行任何复杂计算
        pred = torch.from_numpy(np.load(self.pred_files[idx]).astype(np.float32)).squeeze()
        gt = torch.from_numpy(np.load(self.gt_files[idx]).astype(np.float32)).squeeze()
        return pred, gt

def compute_errors_torch(gt, pred, mask):
    # 向量化误差计算 (GPU)
    valid_pixel_count = mask.sum(dim=[1, 2]).to(torch.float32)
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
    thresh = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    a1 = ((thresh < 1.25) & mask).sum(dim=[1, 2]).float()
    a2 = ((thresh < 1.25 ** 2) & mask).sum(dim=[1, 2]).float()
    a3 = ((thresh < 1.25 ** 3) & mask).sum(dim=[1, 2]).float()

    rmse_map = (gt - pred) ** 2; rmse_map[~mask] = 0
    rmse_s = rmse_map.sum(dim=[1, 2])
    
    rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2; rmse_log_map[~mask] = 0
    rmse_log_s = rmse_log_map.sum(dim=[1, 2])

    abs_rel_map = torch.abs(gt - pred) / gt_c; abs_rel_map[~mask] = 0
    abs_rel_s = abs_rel_map.sum(dim=[1, 2])

    sq_rel_map = ((gt - pred) ** 2) / gt_c; sq_rel_map[~mask] = 0
    sq_rel_s = sq_rel_map.sum(dim=[1, 2])

    return torch.stack([abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count], dim=1)

def compute_metrics_from_sums(sums):
    total_valid = sums[7]
    if total_valid == 0: return np.zeros(7)
    return np.array([
        sums[0]/total_valid, sums[1]/total_valid, np.sqrt(sums[2]/total_valid),
        np.sqrt(sums[3]/total_valid), sums[4]/total_valid, sums[5]/total_valid, sums[6]/total_valid
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
    for pred_b, gt_b in tqdm(dl, desc=pbar_desc, leave=False):
        pred_b, gt_b = pred_b.to(device), gt_b.to(device)
        
        # 1. 动态插值到 GT 尺寸
        if pred_b.shape[-2:] != gt_b.shape[-2:]:
            pred_b = F.interpolate(pred_b.unsqueeze(1), size=gt_b.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        
        # 2. 生成掩码
        mask = (gt_b > MIN_EVAL_DEPTH) & (gt_b < MAX_EVAL_DEPTH)
        
        # 3. 中位数定标 (可选)
        if use_ms:
            B = gt_b.shape[0]
            gt_flat = gt_b.clone(); gt_flat[~mask] = float('nan'); gt_flat = gt_flat.view(B, -1)
            pred_flat = pred_b.clone(); pred_flat[~mask] = float('nan'); pred_flat = pred_flat.view(B, -1)
            
            med_gt = torch.nanmedian(gt_flat, dim=1).values
            med_pred = torch.nanmedian(pred_flat, dim=1).values
            ratio = torch.nan_to_num(med_gt / med_pred, nan=1.0, posinf=1.0, neginf=1.0)
            pred_b = pred_b * ratio.view(B, 1, 1)

        # 4. 裁剪并计算误差
        pred_b.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
        all_errs.append(compute_errors_torch(gt_b, pred_b, mask).cpu().numpy())
            
    return np.concatenate(all_errs, axis=0) if all_errs else None

def format_line(name, m):
    return "{:<50} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} |".format(
        name[-50:], m[0], m[1], m[2], m[3], m[4], m[5], m[6])

# =================================================================================
# 主循环
# =================================================================================
if __name__ == '__main__':
    if not torch.cuda.is_available(): exit("No CUDA device.")
    
    print(f"Batch Inference Started. Total Tasks: {len(TASKS)}")
    
    for i, task in enumerate(TASKS):
        t_name = task['task_name']
        t_out = task['output_file']
        t_ms = task['use_median_scaling']
        t_paths = task['paths'] 
        
        print(f"\n[{i+1}/{len(TASKS)}] Processing Task: {t_name}")
        print(f"  Mode: {'Median Scaling' if t_ms else 'Absolute'}")
        print(f"  Output: {t_out}")
        print(f"  Scenes: {len(t_paths)}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(t_out), exist_ok=True)
        
        results_map = {}
        all_errs_list = []
        
        # 遍历该任务下的每一个字典条目
        for path_dict in t_paths:
            s_name = path_dict['name']
            p_dir = path_dict['pred_dir']
            g_dir = path_dict['gt_dir']

            if not os.path.isdir(p_dir) or not os.path.isdir(g_dir):
                print(f"  [Error] Path not found for scene: {s_name}")
                continue

            errs = evaluate_single_scene(p_dir, g_dir, s_name, t_ms)
            if errs is not None:
                results_map[s_name] = errs
                all_errs_list.append(errs)
        
        # 生成报告
        lines = []
        header = "{:<50} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |".format("Scene", "AbsRel", "SqRel", "RMSE", "RMSElog", "a1", "a2", "a3")
        sep = "-" * len(header)
        lines += [f"Task: {t_name}", f"Date: {datetime.now()}", f"Median Scaling: {t_ms}", "="*100, header, sep]
        
        # 按场景名排序输出
        for s_name in sorted(results_map.keys()):
            s_mean = compute_metrics_from_sums(results_map[s_name].sum(0))
            lines.append(format_line(s_name, s_mean))
            
        # 总平均
        if all_errs_list:
            total_mean = compute_metrics_from_sums(np.concatenate(all_errs_list, 0).sum(0))
            lines += [sep, format_line(">>> OVERALL AVERAGE", total_mean), "="*100]
            
        with open(t_out, 'w') as f: f.write("\n".join(lines))
        print(f"  -> Report saved.")
        
        # 内存清理
        del results_map, all_errs_list
        torch.cuda.empty_cache()
        
    print("\nAll tasks completed.")