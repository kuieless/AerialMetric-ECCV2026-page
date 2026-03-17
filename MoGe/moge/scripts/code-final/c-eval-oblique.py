import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

# ================= 配置区域 =================

SCENE_CATEGORIES = {
    "Rural": [
        "ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output",
        "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled",
        "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled",
        "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled",
        "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output",
        "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled", "BC2", "BC1", "L1"
    ],
    "Natural": [
        "lewis-output", "park5", "park13", "park14", "park10", "park0",
        "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled",
        "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled",
        "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled",
        "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled",
        "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled",
        "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"
    ],
    "City": [
        "yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"
    ],
    "Factory": [
        "factory_scene_1", "factory_scene_2"
    ]
}

# 2. 评估参数
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = False

# ===========================================

def get_category(scene_name):
    for cat, scenes in SCENE_CATEGORIES.items():
        if any(s in scene_name for s in scenes):
            return cat
    return "Uncategorized"

def find_gt_path(pred_path, gt_root_base):
    pred_path = Path(pred_path)
    scene_name = pred_path.parent.name
    file_stem = pred_path.stem 

    gt_scene_dir = Path(gt_root_base) / scene_name
    possible_depth_dirs = [gt_scene_dir / "depth", gt_scene_dir / "depths"]
    
    gt_depth_dir = None
    for d in possible_depth_dirs:
        if d.exists():
            gt_depth_dir = d
            break
    
    if gt_depth_dir is None:
        return None

    candidates = [
        f"{file_stem}.npy",
        f"{file_stem}_depth.npy",
        f"{file_stem}.png",
        f"{file_stem}_depth.png"
    ]

    for cand in candidates:
        target = gt_depth_dir / cand
        if target.exists():
            return target
            
    return None

def align_scale_shift_torch(pred, target, mask):
    # 只使用好像素来计算 Scale
    t_valid = target[mask]
    p_valid = pred[mask]
    
    if len(t_valid) < 10: return pred

    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s, t = solution[0].item(), solution[1].item()
    except:
        s, t = 1.0, 0.0
        
    # 如果算出来的 Scale 也是 NaN (极罕见)，强制设为 1
    if np.isnan(s) or np.isinf(s) or s <= 1e-6:
        s, t = 1.0, 0.0
        
    return pred * s + t

def compute_metrics(gt, pred, mask):
    """
    核心：只对 mask=True 的像素进行计算
    """
    # 1. 提取有效像素 (Flatten)
    # 这一步就把所有 bad pixels 扔掉了
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # 2. 如果有效像素过少，这就没办法了，只能返回 NaN
    if len(gt_valid) < 10:
        return torch.full((9,), float('nan'), device=gt.device)

    # 3. 计算指标 (全部在 1D 向量上操作)
    
    # (a) 基础误差
    mse = ((gt_valid - pred_valid) ** 2).mean()
    rmse = torch.sqrt(mse)
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + 1e-6))
    
    # (b) 准确率 (Threshold)
    # 再次 clamp 防止除以 0 (虽然 mask 应该已经保证了 > 0)
    p_safe = pred_valid.clamp(min=1e-6)
    g_safe = gt_valid.clamp(min=1e-6)
    thresh = torch.maximum((g_safe / p_safe), (p_safe / g_safe))
    
    a1_10 = (thresh < 1.10).float().mean()
    a1_25 = (thresh < 1.25).float().mean()
    a2    = (thresh < 1.25 ** 2).float().mean()
    a3    = (thresh < 1.25 ** 3).float().mean()

    # (c) SI-Log
    # 因为我们已经在外面 mask 掉了 <=0 的值，所以这里可以直接 log
    log_pred = torch.log(p_safe)
    log_gt = torch.log(g_safe)
    log_diff = log_pred - log_gt
    
    # 防止浮点误差导致负数
    var = torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2
    si_log = torch.sqrt(torch.abs(var))

    # (d) N-RMSE
    # 归一化要小心，如果最大值最小值相等，分母为0
    p_min, p_max = pred_valid.min(), pred_valid.max()
    g_min, g_max = gt_valid.min(), gt_valid.max()
    
    range_p = p_max - p_min
    range_g = g_max - g_min
    
    if range_p < 1e-6: range_p = 1.0
    if range_g < 1e-6: range_g = 1.0
    
    p_norm = (pred_valid - p_min) / range_p
    g_norm = (gt_valid - g_min) / range_g
    n_rmse = torch.sqrt(torch.mean((p_norm - g_norm) ** 2))

    # (e) Spearman
    p_cpu = pred_valid.detach().cpu().numpy()
    g_cpu = gt_valid.detach().cpu().numpy()
    
    # 降采样加速
    if len(p_cpu) > 5000:
        idx = np.random.choice(len(p_cpu), 5000, replace=False)
        p_cpu = p_cpu[idx]
        g_cpu = g_cpu[idx]
    
    # 处理常数输出
    if np.std(p_cpu) < 1e-6 or np.std(g_cpu) < 1e-6:
        spearman = 0.0
    else:
        spearman, _ = spearmanr(p_cpu, g_cpu)
        if np.isnan(spearman): spearman = 0.0

    return torch.tensor([abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse, a2, a3])

def run_evaluation(pred_root, gt_root_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_root = Path(pred_root)
    
    print(f"🕵️  正在扫描预测结果: {pred_root}")
    pred_files = sorted(list(pred_root.rglob("*.npy")))
    print(f"✅ 找到 {len(pred_files)} 个预测文件，开始匹配 GT...")

    stats = defaultdict(list)
    category_stats = defaultdict(list)
    overall_stats = []
    
    # 🔥 [新增]: 用于保存逐图详细数据的列表
    per_image_records = []
    
    uncategorized_scenes = set()
    
    pbar = tqdm(pred_files)
    valid_count = 0
    missing_gt_count = 0

    for pred_path in pbar:
        gt_path = find_gt_path(pred_path, gt_root_base)
        if gt_path is None:
            missing_gt_count += 1
            continue

        try:
            pred_np = np.load(pred_path)
            # 即使 npy 有 nan，我们后面也可以 mask 掉，不要在这里 continue
            
            if gt_path.suffix == '.npy':
                gt_np = np.load(gt_path)
            else:
                import cv2
                gt_np = cv2.imread(str(gt_path), -1)
                if gt_np is not None: gt_np = gt_np.astype(np.float32) / 1000.0
            
            if gt_np is None: continue

        except Exception as e:
            print(f"Error reading {pred_path.name}: {e}")
            continue

        pred = torch.from_numpy(pred_np).to(device).squeeze()
        gt = torch.from_numpy(gt_np).to(device).squeeze()

        if pred.shape != gt.shape:
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()

        # ================= 🔥 核心：构建强力 Mask (跳过像素) =================
        # 1. GT 必须在范围内且有效
        mask_gt = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt)
        # 2. Pred 必须是有效值，且必须 > 0 (防止 Log 报错)
        mask_pred = torch.isfinite(pred) & (pred > 0)
        # 3. 合并 Mask
        mask = mask_gt & mask_pred
        
        # 只要还有 10 个有效像素，我们就计算！
        if mask.sum() < 10: 
            # 这种情况下实在没办法，整张图都坏了，只能跳过
            continue

        # Scale 对齐 (只用有效像素计算对齐系数)
        if USE_MEDIAN_SCALING:
            pred = align_scale_shift_torch(pred, gt, mask)
            # 对齐后重新 clamp，防止溢出
            pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

        # 计算指标
        metrics = compute_metrics(gt, pred, mask)
        metrics_np = metrics.cpu().numpy()

        # 如果这一步还是算出 NaN (极低概率)，说明那张图真的完全不可救药
        if np.isnan(metrics_np).any():
            # print(f"⚠️ NaN Metrics: {pred_path.name}")
            continue

        scene_name = pred_path.parent.name
        cat_name = get_category(scene_name)
        file_stem = pred_path.stem
        
        if cat_name == "Uncategorized":
            uncategorized_scenes.add(scene_name)

        stats[scene_name].append(metrics_np)
        category_stats[cat_name].append(metrics_np)
        overall_stats.append(metrics_np)
        
        # 🔥 [新增]: 将单张图的详细信息加入记录列表
        # 根据 compute_metrics 的返回顺序: [abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse, a2, a3]
        per_image_records.append({
            "Category": cat_name,
            "Scene": scene_name,
            "Filename": file_stem,
            "AbsRel": metrics_np[0],
            "RMSE": metrics_np[1],
            "a1.10": metrics_np[2],
            "a1.25": metrics_np[3],
            "SI-Log": metrics_np[4],
            "Spearman": metrics_np[5],
            "N-RMSE": metrics_np[6]
        })
        
        valid_count += 1
        pbar.set_postfix({"Valid": valid_count})

    # ================= 生成报告 =================
    
    print("\n" + "="*120)
    print(f"📊 评估完成! 有效样本: {valid_count}, 缺失GT: {missing_gt_count}")
    print("="*120)

    if uncategorized_scenes:
        print(f"⚠️  未分类场景: {uncategorized_scenes}")
        print("-" * 120)
    
    headers = ["Category/Scene", "N", "AbsRel", "RMSE", "a1.10", "a1.25", "SI-Log", "Spear", "N-RMSE"]
    header_fmt = "{:<22} | {:<4} | {:<7} | {:<7} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}"
    data_fmt   = "{:<22} | {:<4} | {:<7.4f} | {:<7.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f}"
    
    print(header_fmt.format(*headers))
    print("-" * 120)

    # 使用 nanmean 防止偶发的 NaN 污染全局
    
    # 1. Overall
    if overall_stats:
        m = np.nanmean(overall_stats, axis=0)
        print(data_fmt.format("OVERALL", len(overall_stats), *m[:7]))
    print("-" * 120)

    # 2. Category
    for cat in sorted(category_stats.keys()):
        if len(category_stats[cat]) > 0:
            m = np.nanmean(category_stats[cat], axis=0)
            print(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]))
    print("-" * 120)

    # 3. Scene (按 AbsRel 排序)
    scene_avgs = []
    for s, vals in stats.items():
        if len(vals) == 0: continue
        scene_avgs.append((s, len(vals), np.nanmean(vals, axis=0)))
    
    # 按 AbsRel 排序
    scene_avgs.sort(key=lambda x: x[2][0]) 

    print(">>> Top 10 Scenes (Sorted by AbsRel):")
    for s, c, m in scene_avgs[:10]:
         s_short = (s[:20] + '..') if len(s) > 20 else s
         print(data_fmt.format(s_short, c, *m[:7]))

    # 保存 txt
    output_file = pred_root / "Eval_Report_Oblique_Pixel.txt"
    with open(output_file, "w") as f:
        f.write(f"Evaluation Date: {np.datetime64('now')}\n")
        
        if uncategorized_scenes:
            f.write(f"\nUncategorized Scenes: {uncategorized_scenes}\n\n")

        f.write(header_fmt.format(*headers) + "\n")
        f.write("-" * 120 + "\n")
        if overall_stats:
            m = np.nanmean(overall_stats, axis=0)
            f.write(data_fmt.format("OVERALL", len(overall_stats), *m[:7]) + "\n")
        f.write("-" * 120 + "\n")
        for cat in sorted(category_stats.keys()):
            if len(category_stats[cat]) > 0:
                m = np.nanmean(category_stats[cat], axis=0)
                f.write(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]) + "\n")
        f.write("-" * 120 + "\n")
        f.write(">>> All Scenes (Sorted by AbsRel):\n")
        for s, c, m in scene_avgs:
            f.write(data_fmt.format(s, c, *m[:7]) + "\n")
            
    print(f"\n📝 完整报告已保存至: {output_file}")
    
    # 🔥 [新增]: 4. 导出详细逐图数据到 CSV
    if per_image_records:
        df_detailed = pd.DataFrame(per_image_records)
        detailed_output_file = pred_root / "Eval_Report_Oblique_Pixel_Detailed.csv"
        # 存成 CSV 格式，数值保留4位小数，Excel可以直接双击无乱码打开
        df_detailed.to_csv(detailed_output_file, index=False, float_format="%.4f", encoding='utf-8-sig')
        print(f"📊 逐图详细数据已保存至: {detailed_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="/data1/szq/Inference_Results_Base_Model_V2_Original_Size/Val_Extracted/Oblique", help="预测目录")
    parser.add_argument("--gt", default="/data1/szq/Val/Oblique", help="GT根目录")
    args = parser.parse_args()
    
    if not os.path.exists(args.pred):
        print(f"❌ 错误: 目录不存在 {args.pred}")
    else:
        run_evaluation(args.pred, args.gt)

# import os
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from pathlib import Path
# from collections import defaultdict
# from scipy.stats import spearmanr

# # ================= 配置区域 =================

# SCENE_CATEGORIES = {
#     "Rural": [
#         "ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output",
#         "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled",
#         "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled",
#         "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled",
#         "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output",
#         "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled", "BC2", "BC1", "L1"
#     ],
#     "Natural": [
#         "lewis-output", "park5", "park13", "park14", "park10", "park0",
#         "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled",
#         "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled",
#         "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled",
#         "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled",
#         "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled",
#         "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"
#     ],
#     "City": [
#         "yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"
#     ],
#     "Factory": [
#         "factory_scene_1", "factory_scene_2"
#     ]
# }

# # 2. 评估参数
# MIN_DEPTH = 1e-3
# MAX_DEPTH = 400
# USE_MEDIAN_SCALING = False

# # ===========================================

# def get_category(scene_name):
#     for cat, scenes in SCENE_CATEGORIES.items():
#         if any(s in scene_name for s in scenes):
#             return cat
#     return "Uncategorized"

# def find_gt_path(pred_path, gt_root_base):
#     pred_path = Path(pred_path)
#     scene_name = pred_path.parent.name
#     file_stem = pred_path.stem 

#     gt_scene_dir = Path(gt_root_base) / scene_name
#     possible_depth_dirs = [gt_scene_dir / "depth", gt_scene_dir / "depths"]
    
#     gt_depth_dir = None
#     for d in possible_depth_dirs:
#         if d.exists():
#             gt_depth_dir = d
#             break
    
#     if gt_depth_dir is None:
#         return None

#     candidates = [
#         f"{file_stem}.npy",
#         f"{file_stem}_depth.npy",
#         f"{file_stem}.png",
#         f"{file_stem}_depth.png"
#     ]

#     for cand in candidates:
#         target = gt_depth_dir / cand
#         if target.exists():
#             return target
            
#     return None

# def align_scale_shift_torch(pred, target, mask):
#     # 只使用好像素来计算 Scale
#     t_valid = target[mask]
#     p_valid = pred[mask]
    
#     if len(t_valid) < 10: return pred

#     ones = torch.ones_like(p_valid)
#     A = torch.stack([p_valid, ones], dim=1)
    
#     try:
#         solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
#         s, t = solution[0].item(), solution[1].item()
#     except:
#         s, t = 1.0, 0.0
        
#     # 如果算出来的 Scale 也是 NaN (极罕见)，强制设为 1
#     if np.isnan(s) or np.isinf(s) or s <= 1e-6:
#         s, t = 1.0, 0.0
        
#     return pred * s + t

# def compute_metrics(gt, pred, mask):
#     """
#     核心：只对 mask=True 的像素进行计算
#     """
#     # 1. 提取有效像素 (Flatten)
#     # 这一步就把所有 bad pixels 扔掉了
#     gt_valid = gt[mask]
#     pred_valid = pred[mask]
    
#     # 2. 如果有效像素过少，这就没办法了，只能返回 NaN
#     if len(gt_valid) < 10:
#         return torch.full((9,), float('nan'), device=gt.device)

#     # 3. 计算指标 (全部在 1D 向量上操作)
    
#     # (a) 基础误差
#     mse = ((gt_valid - pred_valid) ** 2).mean()
#     rmse = torch.sqrt(mse)
#     abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / (gt_valid + 1e-6))
    
#     # (b) 准确率 (Threshold)
#     # 再次 clamp 防止除以 0 (虽然 mask 应该已经保证了 > 0)
#     p_safe = pred_valid.clamp(min=1e-6)
#     g_safe = gt_valid.clamp(min=1e-6)
#     thresh = torch.maximum((g_safe / p_safe), (p_safe / g_safe))
    
#     a1_10 = (thresh < 1.10).float().mean()
#     a1_25 = (thresh < 1.25).float().mean()
#     a2    = (thresh < 1.25 ** 2).float().mean()
#     a3    = (thresh < 1.25 ** 3).float().mean()

#     # (c) SI-Log
#     # 因为我们已经在外面 mask 掉了 <=0 的值，所以这里可以直接 log
#     log_pred = torch.log(p_safe)
#     log_gt = torch.log(g_safe)
#     log_diff = log_pred - log_gt
    
#     # 防止浮点误差导致负数
#     var = torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2
#     si_log = torch.sqrt(torch.abs(var))

#     # (d) N-RMSE
#     # 归一化要小心，如果最大值最小值相等，分母为0
#     p_min, p_max = pred_valid.min(), pred_valid.max()
#     g_min, g_max = gt_valid.min(), gt_valid.max()
    
#     range_p = p_max - p_min
#     range_g = g_max - g_min
    
#     if range_p < 1e-6: range_p = 1.0
#     if range_g < 1e-6: range_g = 1.0
    
#     p_norm = (pred_valid - p_min) / range_p
#     g_norm = (gt_valid - g_min) / range_g
#     n_rmse = torch.sqrt(torch.mean((p_norm - g_norm) ** 2))

#     # (e) Spearman
#     p_cpu = pred_valid.detach().cpu().numpy()
#     g_cpu = gt_valid.detach().cpu().numpy()
    
#     # 降采样加速
#     if len(p_cpu) > 5000:
#         idx = np.random.choice(len(p_cpu), 5000, replace=False)
#         p_cpu = p_cpu[idx]
#         g_cpu = g_cpu[idx]
    
#     # 处理常数输出
#     if np.std(p_cpu) < 1e-6 or np.std(g_cpu) < 1e-6:
#         spearman = 0.0
#     else:
#         spearman, _ = spearmanr(p_cpu, g_cpu)
#         if np.isnan(spearman): spearman = 0.0

#     return torch.tensor([abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse, a2, a3])

# def run_evaluation(pred_root, gt_root_base):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pred_root = Path(pred_root)
    
#     print(f"🕵️  正在扫描预测结果: {pred_root}")
#     pred_files = sorted(list(pred_root.rglob("*.npy")))
#     print(f"✅ 找到 {len(pred_files)} 个预测文件，开始匹配 GT...")

#     stats = defaultdict(list)
#     category_stats = defaultdict(list)
#     overall_stats = []
    
#     uncategorized_scenes = set()
    
#     pbar = tqdm(pred_files)
#     valid_count = 0
#     missing_gt_count = 0

#     for pred_path in pbar:
#         gt_path = find_gt_path(pred_path, gt_root_base)
#         if gt_path is None:
#             missing_gt_count += 1
#             continue

#         try:
#             pred_np = np.load(pred_path)
#             # 即使 npy 有 nan，我们后面也可以 mask 掉，不要在这里 continue
            
#             if gt_path.suffix == '.npy':
#                 gt_np = np.load(gt_path)
#             else:
#                 import cv2
#                 gt_np = cv2.imread(str(gt_path), -1)
#                 if gt_np is not None: gt_np = gt_np.astype(np.float32) / 1000.0
            
#             if gt_np is None: continue

#         except Exception as e:
#             print(f"Error reading {pred_path.name}: {e}")
#             continue

#         pred = torch.from_numpy(pred_np).to(device).squeeze()
#         gt = torch.from_numpy(gt_np).to(device).squeeze()

#         if pred.shape != gt.shape:
#             pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()

#         # ================= 🔥 核心：构建强力 Mask (跳过像素) =================
#         # 1. GT 必须在范围内且有效
#         mask_gt = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt)
#         # 2. Pred 必须是有效值，且必须 > 0 (防止 Log 报错)
#         mask_pred = torch.isfinite(pred) & (pred > 0)
#         # 3. 合并 Mask
#         mask = mask_gt & mask_pred
        
#         # 只要还有 10 个有效像素，我们就计算！
#         if mask.sum() < 10: 
#             # 这种情况下实在没办法，整张图都坏了，只能跳过
#             continue

#         # Scale 对齐 (只用有效像素计算对齐系数)
#         if USE_MEDIAN_SCALING:
#             pred = align_scale_shift_torch(pred, gt, mask)
#             # 对齐后重新 clamp，防止溢出
#             pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

#         # 计算指标
#         metrics = compute_metrics(gt, pred, mask)
#         metrics_np = metrics.cpu().numpy()

#         # 如果这一步还是算出 NaN (极低概率)，说明那张图真的完全不可救药
#         if np.isnan(metrics_np).any():
#             # print(f"⚠️ NaN Metrics: {pred_path.name}")
#             continue

#         scene_name = pred_path.parent.name
#         cat_name = get_category(scene_name)
        
#         if cat_name == "Uncategorized":
#             uncategorized_scenes.add(scene_name)

#         stats[scene_name].append(metrics_np)
#         category_stats[cat_name].append(metrics_np)
#         overall_stats.append(metrics_np)
        
#         valid_count += 1
#         pbar.set_postfix({"Valid": valid_count})

#     # ================= 生成报告 =================
    
#     print("\n" + "="*120)
#     print(f"📊 评估完成! 有效样本: {valid_count}, 缺失GT: {missing_gt_count}")
#     print("="*120)

#     if uncategorized_scenes:
#         print(f"⚠️  未分类场景: {uncategorized_scenes}")
#         print("-" * 120)
    
#     headers = ["Category/Scene", "N", "AbsRel", "RMSE", "a1.10", "a1.25", "SI-Log", "Spear", "N-RMSE"]
#     header_fmt = "{:<22} | {:<4} | {:<7} | {:<7} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}"
#     data_fmt   = "{:<22} | {:<4} | {:<7.4f} | {:<7.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f}"
    
#     print(header_fmt.format(*headers))
#     print("-" * 120)

#     # 使用 nanmean 防止偶发的 NaN 污染全局
    
#     # 1. Overall
#     if overall_stats:
#         m = np.nanmean(overall_stats, axis=0)
#         print(data_fmt.format("OVERALL", len(overall_stats), *m[:7]))
#     print("-" * 120)

#     # 2. Category
#     for cat in sorted(category_stats.keys()):
#         if len(category_stats[cat]) > 0:
#             m = np.nanmean(category_stats[cat], axis=0)
#             print(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]))
#     print("-" * 120)

#     # 3. Scene (按 AbsRel 排序)
#     scene_avgs = []
#     for s, vals in stats.items():
#         if len(vals) == 0: continue
#         scene_avgs.append((s, len(vals), np.nanmean(vals, axis=0)))
    
#     # 按 AbsRel 排序
#     scene_avgs.sort(key=lambda x: x[2][0]) 

#     print(">>> Top 10 Scenes (Sorted by AbsRel):")
#     for s, c, m in scene_avgs[:10]:
#          s_short = (s[:20] + '..') if len(s) > 20 else s
#          print(data_fmt.format(s_short, c, *m[:7]))

#     # 保存
#     output_file = pred_root / "Eval_Report_Oblique_Pixel.txt"
#     with open(output_file, "w") as f:
#         f.write(f"Evaluation Date: {np.datetime64('now')}\n")
        
#         if uncategorized_scenes:
#             f.write(f"\nUncategorized Scenes: {uncategorized_scenes}\n\n")

#         f.write(header_fmt.format(*headers) + "\n")
#         f.write("-" * 120 + "\n")
#         if overall_stats:
#             m = np.nanmean(overall_stats, axis=0)
#             f.write(data_fmt.format("OVERALL", len(overall_stats), *m[:7]) + "\n")
#         f.write("-" * 120 + "\n")
#         for cat in sorted(category_stats.keys()):
#             if len(category_stats[cat]) > 0:
#                 m = np.nanmean(category_stats[cat], axis=0)
#                 f.write(data_fmt.format(f"[CAT] {cat}", len(category_stats[cat]), *m[:7]) + "\n")
#         f.write("-" * 120 + "\n")
#         f.write(">>> All Scenes (Sorted by AbsRel):\n")
#         for s, c, m in scene_avgs:
#             f.write(data_fmt.format(s, c, *m[:7]) + "\n")
            
#     print(f"\n📝 完整报告已保存至: {output_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pred", default="/data1/szq/Inference_Results_Base_Model_V2_Original_Size/Val_Extracted/Oblique", help="预测目录")
#     parser.add_argument("--gt", default="/data1/szq/Val/Oblique", help="GT根目录")
#     args = parser.parse_args()
    
#     if not os.path.exists(args.pred):
#         print(f"❌ 错误: 目录不存在 {args.pred}")
#     else:
#         run_evaluation(args.pred, args.gt)