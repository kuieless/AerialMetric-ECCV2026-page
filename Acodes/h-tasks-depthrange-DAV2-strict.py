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
# 1. 配置区域
# =================================================================================

# -------------------------------------------------------------------------
# 数据路径定义 (请在此处填入您的实际路径列表)
# -------------------------------------------------------------------------
# 示例: PATHS_M3D_L = [ {'name': '...', 'pred_dir': '...', 'gt_dir': '...'}, ... ]

PATHS_M3D_L = [
    # {
    #     'name': 'D1',
    #     'pred_dir': '/home/data1/szq/Megadepth/metric3D/D1/infer3/Infer3-l/D1-images/depth_npy/D1-images',
    #     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/D1-npy'
    # },
    # 请在这里补充您完整的列表...

    {
        'name': 'lower',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/lower',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/lower-npy'
    },
    {
        'name': 'sziit',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/sziit',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/sziit-npy'
    },
    {
        'name': 'dj3',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/dj3',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/dj3-npy'
    },
    {
        'name': 'hsd1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/hsd1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/hsd1-npy'
    },
    {
        'name': 'xg5',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/xg5',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/xg5-npy'
    },
    {
        'name': 'town1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/town1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town1-npy'
    },
    {
        'name': 'town2',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/town2',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town2-npy'
    },
    {
        'name': 'town3',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/town3',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town3-npy'
    },
    {
        'name': 'yingrenshi1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/yingrenshi1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yingrenshi1-npy'
    },
    {
        'name': 'yingrenshi2',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/yingrenshi2',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yingrenshi2-npy'
    },
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yuehai-npy'
    },
    {
        'name': 'D1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/D1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/D1-npy'
    },
    {
        'name': 'R1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/R1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-npy'
    },
    {
        'name': 'R1-PHD',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/R1-PHD',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-PHD-npy'
    },
    {
        'name': 'bellus',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/bellus',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/bellus-npy'
    },
    {
        'name': 'brighton-beach',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/brighton-beach',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/brighton-beach-npy'
    },
    {
        'name': 'ODM1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/ODM1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/ODM1-npy'
    },
    {
        'name': 'park13',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/park13',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/park13-npy'
    },
    {
        'name': 'seneca',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/seneca',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/seneca-npy'
    },


]

# 为了演示，如果上面的列表为空，脚本会跳过。请确保填入数据。

TASKS = [
    # {
    #     "task_name": "M3D-G-Absolutedepthrange",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/M3D/Infer3-g.txt",
    #     "use_median_scaling": False,
    #     "paths": PATHS_M3D_G
    # },
    {
        "task_name": "DAV2-l-Absolutedepthrange-Strict",
        "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/DAV2/Infer6-l-Strict.txt",
        "use_median_scaling": False, # 是否使用中值定标
        "paths": PATHS_M3D_L
    },
    {
        "task_name": "DAV2-l-Absolutedepthrange-Strict-MS",
        "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/DAV2/Infer6-l-Strict-MS.txt",
        "use_median_scaling": True, 
        "paths": PATHS_M3D_L
    },
]

# -------------------------------------------------------------------------
# 深度分段区间 (左闭右开: [min, max))
# -------------------------------------------------------------------------
DEPTH_BINS = [
    (0, 50),
    (50, 120),
    (120, 250),
    (250, 400),
    (400, 500)
]
BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]

# 系统设置
BATCH_SIZE = 16  # 如果显存够大可以调大
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 500

# =================================================================================
# 2. 数据集定义
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
            # 允许空列表，但在主循环中会报错跳过
            pass 

    def __len__(self): return len(self.pred_files)
    
    def __getitem__(self, idx):
        # 读取 .npy 文件
        pred = torch.from_numpy(np.load(self.pred_files[idx]).astype(np.float32)).squeeze()
        gt = torch.from_numpy(np.load(self.gt_files[idx]).astype(np.float32)).squeeze()
        return pred, gt

# =================================================================================
# 3. 核心指标计算 (含严格指标)
# =================================================================================

def compute_errors_torch_bins(gt, pred, valid_mask):
    """
    计算所有分段的指标，包含针对航拍优化的严格阈值。
    返回形状: [Batch, Num_Bins + 1, 10]
    """
    # 1. 预处理
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
    # 2. 预计算 Error Maps
    rmse_map = (gt - pred) ** 2
    rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
    abs_rel_map = torch.abs(gt - pred) / gt_c
    sq_rel_map = ((gt - pred) ** 2) / gt_c
    
    # 计算比率 max(gt/pred, pred/gt)
    thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
    # 定义所有要统计的区间 (Bins + Overall)
    all_ranges = DEPTH_BINS + [("Overall", "Overall")] 
    
    batch_results = []
    
    for (b_min, b_max) in all_ranges:
        # 3. 生成 Mask
        if b_min == "Overall":
            current_mask = valid_mask
        else:
            current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
            
        mask_f = current_mask.float()
        valid_pixel_count = mask_f.sum(dim=[1, 2]) # [Batch]
        
        # 4. 聚合计算
        # --- 标准指标 ---
        # a1 (delta < 1.25)
        a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
        # a2 (delta < 1.25^2 = 1.5625)
        a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
        # a3 (delta < 1.25^3 = 1.953)
        a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
        # --- 新增：航拍严格指标 ---
        # a_strict_1 (delta < 1.10) -> 10% 误差以内
        a_strict_1 = ((thresh_val < 1.10) & current_mask).sum(dim=[1, 2]).float()
        # a_strict_2 (delta < 1.05) -> 5% 误差以内 (高精度)
        a_strict_2 = ((thresh_val < 1.05) & current_mask).sum(dim=[1, 2]).float()
        
        # 误差求和
        rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
        rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
        abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
        sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])
        
        # Stack 顺序:
        # 0:AbsRel, 1:SqRel, 2:RMSE, 3:RMSElog, 4:a1(1.25), 5:a2, 6:a3, 7:Count, 8:a(1.10), 9:a(1.05)
        bin_res = torch.stack([
            abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, 
            a1, a2, a3, valid_pixel_count, 
            a_strict_1, a_strict_2
        ], dim=1)
        batch_results.append(bin_res)
        
    # [Batch, Num_Bins+1, 10]
    return torch.stack(batch_results, dim=1)

def compute_metrics_from_sums(sums):
    """从累加值计算平均值"""
    total_valid = sums[7]
    if total_valid <= 0: 
        return np.zeros(10)
    
    return np.array([
        sums[0]/total_valid, # AbsRel
        sums[1]/total_valid, # SqRel
        np.sqrt(sums[2]/total_valid), # RMSE
        sums[3]/total_valid, # RMSElog
        sums[4]/total_valid, # a1 (1.25)
        sums[5]/total_valid, # a2
        sums[6]/total_valid, # a3
        sums[7],             # Count
        sums[8]/total_valid, # a_1.10 (New)
        sums[9]/total_valid  # a_1.05 (New)
    ])

def format_line(name, m, indent=0):
    """格式化输出行，包含严格指标"""
    sp = " " * indent
    m = np.nan_to_num(m) 
    # 显示: Name | AbsRel | RMSE | 1.25 | 1.10 | 1.05
    # 如果您想看 SqRel 或 1.25^2/3，可以自行调整
    return "{:<50} | {:>8.4f} | {:>8.2f} | {:>8.4f} | {:>8.4f} | {:>8.4f} |".format(
        sp + name[-50+indent:], 
        m[0],   # AbsRel
        m[2],   # RMSE
        m[4],   # d < 1.25
        m[8],   # d < 1.10 (Strict)
        m[9]    # d < 1.05 (Very Strict)
    )

def evaluate_single_scene(pred_dir, gt_dir, scene_name, use_ms):
    device = torch.device("cuda")
    
    # 数据加载
    ds = PairedNpyDataset(pred_dir, gt_dir)
    if len(ds) == 0:
        print(f"  [Skipped] Empty/Mismatch: {scene_name}")
        return None

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    all_errs = []
    
    for pred_b, gt_b in tqdm(dl, desc=f"  -> {scene_name}", leave=False):
        pred_b, gt_b = pred_b.to(device), gt_b.to(device)
        
        # 对齐尺寸 (如果有微小差异)
        if pred_b.shape[-2:] != gt_b.shape[-2:]:
            pred_b = F.interpolate(pred_b.unsqueeze(1), size=gt_b.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        
        # 生成有效 Mask
        mask = (gt_b > MIN_EVAL_DEPTH) & (gt_b < MAX_EVAL_DEPTH)
        
        # Median Scaling (定标)
        if use_ms:
            B = gt_b.shape[0]
            gt_flat = gt_b.clone(); gt_flat[~mask] = float('nan'); gt_flat = gt_flat.view(B, -1)
            pred_flat = pred_b.clone(); pred_flat[~mask] = float('nan'); pred_flat = pred_flat.view(B, -1)
            
            med_gt = torch.nanmedian(gt_flat, dim=1).values
            med_pred = torch.nanmedian(pred_flat, dim=1).values
            ratio = torch.nan_to_num(med_gt / med_pred, nan=1.0, posinf=1.0, neginf=1.0)
            
            pred_b = pred_b * ratio.view(B, 1, 1)

        pred_b.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
        
        # 计算误差
        batch_res = compute_errors_torch_bins(gt_b, pred_b, mask).cpu().numpy()
        all_errs.append(batch_res)
            
    if not all_errs: return None
    return np.concatenate(all_errs, axis=0)

# =================================================================================
# 4. 主程序
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
        
        os.makedirs(os.path.dirname(t_out), exist_ok=True)
        
        results_map = {}
        all_errs_list = []
        
        # --- 遍历所有场景 ---
        for path_dict in t_paths:
            s_name = path_dict['name']
            p_dir = path_dict['pred_dir']
            g_dir = path_dict['gt_dir']

            if not os.path.isdir(p_dir) or not os.path.isdir(g_dir):
                print(f"  [Error] Path not found for scene: {s_name}")
                continue

            scene_errs = evaluate_single_scene(p_dir, g_dir, s_name, t_ms)
            
            if scene_errs is not None:
                results_map[s_name] = scene_errs
                all_errs_list.append(scene_errs)
        
        # --- 生成报告 ---
        if not results_map:
            print("  No valid results computed.")
            continue

        lines = []
        # Header 重点突出了 AbsRel, RMSE 以及新增的 strict delta
        header = "{:<50} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |".format(
            "Scene / Depth Bin", "AbsRel", "RMSE(m)", "d<1.25", "d<1.10", "d<1.05")
        sep = "-" * len(header)
        
        lines += [
            f"Task: {t_name}", 
            f"Date: {datetime.now()}", 
            f"Median Scaling: {t_ms}", 
            f"Note: d<1.10 (Strict, 10% err), d<1.05 (Very Strict, 5% err)",
            "="*100, 
            header, 
            sep
        ]
        
        # 1. 逐场景报告
        for s_name in sorted(results_map.keys()):
            s_data = results_map[s_name]
            s_sums = s_data.sum(axis=0)
            
            # Overall
            overall_mean = compute_metrics_from_sums(s_sums[-1])
            lines.append(format_line(f"> {s_name} (All)", overall_mean))
            
            # Depth Bins
            for bin_idx, bin_label in enumerate(BIN_LABELS):
                bin_mean = compute_metrics_from_sums(s_sums[bin_idx])
                lines.append(format_line(f"   [{bin_label}]", bin_mean, indent=3))
            
            lines.append(sep)
            
        # 2. 数据集总平均
        if all_errs_list:
            total_data = np.concatenate(all_errs_list, axis=0)
            total_sums = total_data.sum(axis=0)
            
            lines.append("="*100)
            lines.append(">>> DATASET TOTAL AVERAGE <<<")
            lines.append(sep)
            
            total_overall_mean = compute_metrics_from_sums(total_sums[-1])
            lines.append(format_line("TOTAL (All Depths)", total_overall_mean))
            
            for bin_idx, bin_label in enumerate(BIN_LABELS):
                bin_mean = compute_metrics_from_sums(total_sums[bin_idx])
                lines.append(format_line(f"TOTAL [{bin_label}]", bin_mean))
            
            lines.append("="*100)
            
        # 写入文件
        with open(t_out, 'w') as f: f.write("\n".join(lines))
        print(f"  -> Report saved to {t_out}")
        
        # 清理内存
        del results_map, all_errs_list
        torch.cuda.empty_cache()
        
    print("\nAll tasks completed.")