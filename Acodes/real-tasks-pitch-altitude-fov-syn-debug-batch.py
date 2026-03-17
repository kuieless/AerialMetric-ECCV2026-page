

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import warnings
from scipy.stats import spearmanr  # <--- 新增引用

# ================= 配置区域 =================

# 1. 输入数据
MASTER_CSV = "/home/data1/szq/Megadepth/Benchmark-final2/final_merged.csv"
PRED_ROOT = "/home/data1/szq/Megadepth/becnmarkdata2/Val-Results-moge2-500step-out"
VAL_GT_ROOTS = [
    '/home/data1/szq/Megadepth/Benchmark-final2/Val',
]

# 2. 评估参数
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 400
USE_MEDIAN_SCALING = False  # 开启 Least Squares Scale+Shift 对齐
BATCH_SIZE = 1 
NUM_WORKERS = 12
FOCAL_STEP = 400

# 3. 深度分段
DEPTH_BINS = [
    (0, 50), (50, 120), (120, 250), (250, 400), (400, 500)
]
BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]
ALL_LABELS = BIN_LABELS + ["Overall"]

# ================= 核心逻辑 =================

warnings.filterwarnings("ignore")

def align_scale_shift_torch(pred, target, mask):
    """鲁棒的最小二乘法对齐"""
    safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    t_valid = target[safe_mask]
    p_valid = pred[safe_mask]
    
    if len(t_valid) < 10:
        if len(t_valid) > 0:
            ratio = torch.median(t_valid) / (torch.median(p_valid) + 1e-8)
            return pred * ratio
        else:
            return pred 

    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    
    s, t = 1.0, 0.0
    success = False
    
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s = solution[0].item()
        t = solution[1].item()
        if np.isnan(s) or np.isnan(t) or np.isinf(s) or np.isinf(t): raise ValueError()
        if s <= 1e-4: raise ValueError()
        success = True
    except Exception:
        success = False
        
    if not success:
        med_t = torch.median(t_valid)
        med_p = torch.median(p_valid)
        if med_p > 1e-6:
            s = (med_t / med_p).item()
            t = 0.0
        else:
            s, t = 1.0, 0.0

    return pred * s + t

def compute_errors_torch_bins(gt, pred, valid_mask):
    """
    计算所有指标，包含新增的 SI-Log, Spearman, NormRMSE
    注意：为了兼容 Batch 处理，这里返回的是 [Batch, Range, Metrics]
    """
    # 确保维度一致
    if pred.dim() == 3 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
    if gt.dim() == 3 and gt.shape[-1] == 1: gt = gt.squeeze(-1)
    
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
    # === 1. Pixel-wise Metrics (像素级指标，需按像素数累加) ===
    rmse_map = (gt - pred) ** 2
    rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
    abs_rel_map = torch.abs(gt - pred) / gt_c
    sq_rel_map = ((gt - pred) ** 2) / gt_c
    thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
    all_ranges = DEPTH_BINS + [("Overall", "Overall")]
    batch_results = []
    
    for (b_min, b_max) in all_ranges:
        if b_min == "Overall":
            current_mask = valid_mask
        else:
            current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
            
        mask_f = current_mask.float()
        valid_pixel_count = mask_f.sum(dim=[1, 2]) # [Batch_Size]
        
        # --- 传统指标 (Sums) ---
        a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
        a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
        a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
        rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
        rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
        abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
        sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])

        # === 2. Structural Metrics (结构性指标，按图像平均，不按像素累加) ===
        # 我们在这里计算单张图的分数，后续聚合时除以 count_img
        
        si_log_list = []
        spearman_list = []
        norm_rmse_list = []
        
        # 由于这些指标无法简单通过矩阵运算并行处理(特别是Spearman)，我们需要循环Batch
        # 但因为这里 BATCH_SIZE=1，循环只有一次
        for b in range(gt.shape[0]):
            m = current_mask[b]
            if m.sum() < 10:
                si_log_list.append(0.0)
                spearman_list.append(0.0)
                norm_rmse_list.append(0.0)
                continue
                
            g_valid = gt_c[b][m]
            p_valid = pred_c[b][m]

            # A. SI-Log (Scale Invariant Log Error)
            # Formula: Mean(diff^2) - (Mean(diff))^2
            log_diff = torch.log(g_valid) - torch.log(p_valid)
            # * 100 是为了让数值好看一点，通常论文里是 x100 或者直接显示小数
            # 这里我们保持原始小数，越小越好
            si_log_val = torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2 + 1e-8)
            si_log_list.append(si_log_val.item())
            
            # B. Spearman (Rank Correlation)
            # 需要转到 CPU 计算，且如果点太多，为了速度可以采样
            p_np = p_valid.detach().cpu().numpy()
            g_np = g_valid.detach().cpu().numpy()
            
            # 采样优化：如果点超过 5000 个，随机采 5000 个算 (极大加速)
            if len(p_np) > 5000:
                idx = np.random.choice(len(p_np), 5000, replace=False)
                p_np = p_np[idx]
                g_np = g_np[idx]
            
            # 处理完全平坦的情况避免 NaN
            if np.std(p_np) < 1e-6 or np.std(g_np) < 1e-6:
                spearman_val = 0.0
            else:
                spearman_val, _ = spearmanr(p_np, g_np)
                # 处理 spearman 可能返回 nan 的情况
                if np.isnan(spearman_val): spearman_val = 0.0
            
            spearman_list.append(spearman_val)
            
            # C. Normalized RMSE (0-1 Space)
            # (x - min) / (max - min)
            p_norm = (p_valid - p_valid.min()) / (p_valid.max() - p_valid.min() + 1e-8)
            g_norm = (g_valid - g_valid.min()) / (g_valid.max() - g_valid.min() + 1e-8)
            norm_rmse_val = torch.sqrt(torch.mean((p_norm - g_norm)**2))
            norm_rmse_list.append(norm_rmse_val.item())

        si_log_t = torch.tensor(si_log_list, device=gt.device)
        spearman_t = torch.tensor(spearman_list, device=gt.device)
        norm_rmse_t = torch.tensor(norm_rmse_list, device=gt.device)
        
        # 统计有效图像数量 (如果该分段该图没有足够像素，则不算作有效图像)
        valid_img_count = (valid_pixel_count > 10).float()

        # 组合结果 [Batch, 11]
        # 前8个是像素累加值，后3个是单图评分，最后1个是有效图像标记
        bin_res = torch.stack([
            abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count,
            si_log_t, spearman_t, norm_rmse_t, valid_img_count
        ], dim=1)
        batch_results.append(bin_res)
        
    return torch.stack(batch_results, dim=1)

def compute_metrics_from_sums(sums):
    """
    根据累加结果计算最终平均值
    sums 结构: [0:abs, 1:sq, 2:rmse, 3:log, 4:a1, 5:a2, 6:a3, 7:px_count, 
               8:si_log_sum, 9:spear_sum, 10:norm_rmse_sum, 11:img_count]
    """
    total_valid_pixels = sums[7]
    total_valid_imgs = sums[11]
    
    if total_valid_pixels <= 0: return np.zeros(11)
    
    # 1. 像素级指标 (除以总像素数)
    pixel_metrics = np.array([
        sums[0]/total_valid_pixels,           # abs_rel
        sums[1]/total_valid_pixels,           # sq_rel
        np.sqrt(sums[2]/total_valid_pixels),  # rmse
        np.sqrt(sums[3]/total_valid_pixels),  # rmse_log
        sums[4]/total_valid_pixels,           # a1
        sums[5]/total_valid_pixels,           # a2
        sums[6]/total_valid_pixels            # a3
    ])
    
    # 2. 图像级指标 (除以有效图像数)
    if total_valid_imgs > 0:
        img_metrics = np.array([
            sums[8]/total_valid_imgs,   # SI-Log
            sums[9]/total_valid_imgs,   # Spearman
            sums[10]/total_valid_imgs   # Norm-RMSE
        ])
    else:
        img_metrics = np.zeros(3)
        
    return np.concatenate([pixel_metrics, img_metrics])

class CSVIndexDataset(Dataset):
    def __init__(self, csv_path, pred_root, gt_roots):
        self.df = pd.read_csv(csv_path)
        self.val_df = self.df[self.df['Split'] == 'val'].reset_index(drop=True)
        self.pred_root = Path(pred_root)
        self.gt_roots = [Path(p) for p in gt_roots]
        print(f"Dataset initialized. Val samples: {len(self.val_df)}")

    def __len__(self): return len(self.val_df)
    
    def find_file(self, roots, scene, sub, stems):
        for root in roots:
            base = root / scene / sub
            if not base.exists(): continue
            for stem in stems:
                candidates = [f"{stem}.npy", f"{stem}_depth.npy", f"{stem}_Depth.npy"]
                for cand in candidates:
                    f = base / cand
                    if f.exists(): return f
                try:
                    lower_map = {f.lower(): f for f in os.listdir(base)}
                    for cand in candidates:
                        if cand.lower() in lower_map: return base / lower_map[cand.lower()]
                except: pass
        return None

    def __getitem__(self, idx):
        row = self.val_df.iloc[idx]
        scene = str(row['Scene_Name'])
        renamed = str(row['Renamed_Image'])
        original = str(row['Original_Filename'])
        stems = []
        if pd.notna(renamed): stems.append(os.path.splitext(renamed)[0])
        if pd.notna(original): stems.append(os.path.splitext(original)[0])
        
        gt_path = self.find_file(self.gt_roots, scene, "depth", stems)
        pred_path = self.find_file([self.pred_root], scene, "", stems)
        
        if gt_path is None or pred_path is None: return None 
            
        try:
            gt = np.load(gt_path).astype(np.float32)
            pred = np.load(pred_path).astype(np.float32)
            focal_val = row.get('FocalLength_New', row.get('FocalLength(px)', 0))
            meta = {
                'scene': scene,
                'pitch': row.get('Visual_Pitch(deg)', 0),
                'focal': focal_val
            }
            return torch.from_numpy(pred), torch.from_numpy(gt), meta
        except: return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    preds = [b[0] for b in batch]
    gts = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return preds, gts, metas

# ================= 辅助函数 =================

def format_line(name, m, count_img=0, count_scn=0, indent=0):
    sp = " " * indent
    m = np.nan_to_num(m)
    # 增加三个新列的打印 (SI-Log, Spear, N-RMSE)
    return "{:<20} | {:>4} | {:>4} | {:>6.4f} | {:>6.2f} | {:>6.4f} | {:>6.4f} | {:>6.4f} | {:>6.4f} || {:>6.4f} | {:>6.4f} | {:>6.4f} |".format(
        sp + str(name)[-20+indent:], 
        count_img, count_scn,
        m[0], m[2], m[3], m[4], m[5], m[6], # AbsRel, RMSE, RMSElog, a1, a2, a3
        m[7], m[8], m[9] # SI-Log, Spearman, Norm-RMSE
    )

def format_header():
    # 调整表头，去掉了 SqRel 以节省空间给新指标
    return "{:<20} | {:>4} | {:>4} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} || {:>6} | {:>6} | {:>6} |".format(
        "Category", "Img#", "Scn#", "AbsRel", "RMSE", "RMSElg", "a1", "a2", "a3", "SI-Log", "Spear", "N-RMSE")

# ================= 主流程 =================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CSVIndexDataset(MASTER_CSV, PRED_ROOT, VAL_GT_ROOTS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # 累加器初始化：现在长度为 12 (8个旧的 + 3个新的 + 1个图片计数)
    acc_overall = torch.zeros(len(ALL_LABELS), 12).to(device)
    acc_pitch = {}
    acc_focal = {}
    acc_scene = {}
    
    count_pitch = {}
    count_focal = {}
    count_scene = {}
    total_imgs = 0
    
    print("🚀 开始全量评估 (含 Spearman & SI-Log)...")
    
    for batch in tqdm(loader):
        if batch is None: continue
        preds, gts, metas = batch
        
        for i in range(len(preds)):
            pred = preds[i].to(device)
            gt = gts[i].to(device)
            meta = metas[i]
            
            if pred.dim() == 3 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
            if gt.dim() == 3 and gt.shape[-1] == 1: gt = gt.squeeze(-1)
            if pred.shape != gt.shape:
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()
            
            mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH)
            if mask.sum() < 10: continue
            
            # 1. 鲁棒对齐 (LS Scale+Shift)
            if USE_MEDIAN_SCALING:
                pred = align_scale_shift_torch(pred, gt, mask)
            
            pred.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
            
            # 2. 计算所有指标
            batch_sums = compute_errors_torch_bins(gt.unsqueeze(0), pred.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            
            total_imgs += 1
            acc_overall += batch_sums
            
            def update_stats(acc_dict, count_dict, key, vals, scene_name):
                if key not in acc_dict:
                    acc_dict[key] = torch.zeros_like(vals)
                    count_dict[key] = {'img': 0, 'scn': set()}
                acc_dict[key] += vals
                count_dict[key]['img'] += 1
                count_dict[key]['scn'].add(scene_name)

            # Pitch
            p_val = float(meta['pitch'])
            if p_val < -75: p_key = "-90to-75"
            elif p_val < -60: p_key = "-75to-60"
            elif p_val < -45: p_key = "-60to-45"
            elif p_val < -30: p_key = "-45to-30"
            else: p_key = "-30to-0"
            update_stats(acc_pitch, count_pitch, p_key, batch_sums, meta['scene'])
            
            # Focal
            f_val = meta['focal']
            f_start = int(f_val // FOCAL_STEP) * FOCAL_STEP
            f_key = f"{f_start}-{f_start + FOCAL_STEP}"
            update_stats(acc_focal, count_focal, f_key, batch_sums, meta['scene'])
            
            # Scene
            update_stats(acc_scene, count_scene, meta['scene'], batch_sums, meta['scene'])

    # ================= 生成报告 =================
    
    lines = []
    header = format_header()
    sep = "-" * len(header)
    lines += ["EVALUATION REPORT (Robust Aligned + Structural Metrics)", f"Date: {datetime.now()}", "="*130, header, sep]
    
    ov_res = compute_metrics_from_sums(acc_overall[-1].cpu().numpy())
    all_scenes = set()
    for v in count_scene.values(): all_scenes.update(v['scn'])
    lines.append(format_line("OVERALL", ov_res, total_imgs, len(all_scenes)))
    lines.append(sep)
    
    lines.append(">>> BY PITCH (Angle < 0)")
    pitch_order = ["-90to-75", "-75to-60", "-60to-45", "-45to-30", "-30to-0"]
    sorted_keys = sorted(acc_pitch.keys(), key=lambda x: pitch_order.index(x) if x in pitch_order else 999)
    for k in sorted_keys:
        res = compute_metrics_from_sums(acc_pitch[k][-1].cpu().numpy())
        cnt = count_pitch[k]
        lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
    lines.append(sep)
    
    lines.append(f">>> BY FOCAL (Step {FOCAL_STEP})")
    sorted_focal = sorted(acc_focal.keys(), key=lambda x: int(x.split('-')[0]))
    for k in sorted_focal:
        res = compute_metrics_from_sums(acc_focal[k][-1].cpu().numpy())
        cnt = count_focal[k]
        lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
    lines.append(sep)
    
    lines.append(">>> ALL SCENES (Sorted by Spearman: Best -> Worst)")
    scene_list = []
    for k in acc_scene.keys():
        res = compute_metrics_from_sums(acc_scene[k][-1].cpu().numpy())
        cnt = count_scene[k]
        scene_list.append((k, res, cnt))
    # 按照 Spearman 排序 (索引 8)
    scene_list.sort(key=lambda x: x[1][8], reverse=True)
    for k, res, cnt in scene_list:
        lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
    
    lines.append("="*130)
    
    report_text = "\n".join(lines)
    print(report_text)
    
    out_file = os.path.join(PRED_ROOT, "Final_Report_Structural.txt")
    with open(out_file, "w") as f: f.write(report_text)
    print(f"\n✅ 报告已保存至: {out_file}")