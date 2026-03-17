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

# 1. 数据集名称到 CSV 路径的映射
#    注意：Key 必须是你文件夹的名字 (Cleaned_Dataset_...)
CSV_MAP = {
    "Cleaned_Dataset_Campus":  "/data1/szq/Val/Bench/final_dataset_campus.csv",
    "Cleaned_Dataset_Factory": "/data1/szq/Val/Bench/final_dataset_factory.csv",
    "Cleaned_Dataset_Farm":    "/data1/szq/Val/Bench/final_dataset_farm.csv",
    "Cleaned_Dataset_Gress":   "/data1/szq/Val/Bench/final_dataset_grass.csv" # 注意这里的 Gress/grass 拼写差异
}

# 2. 评估参数 (保持与 Oblique 一致)
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = False

# ===========================================

def load_csv_metadata(csv_path):
    """读取 CSV 并建立索引：Filename_Stem -> {pitch, height}"""
    try:
        # 尝试 UTF-8，如果报错尝试 GBK (中文 CSV 常见问题)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        meta_dict = {}
        # 必须确保列名存在，做一些容错处理
        target_col = '匹配到的参考图(Target)'
        pitch_col = '实际Pitch' 
        height_col = '相对高度(自动)'

        if target_col not in df.columns:
            print(f"⚠️ Warning: CSV {csv_path} 缺少 '{target_col}' 列")
            return {}

        for _, row in df.iterrows():
            target_name = str(row[target_col]) # e.g., "abcde.JPG"
            stem = os.path.splitext(target_name)[0] # "abcde"
            
            pitch = row[pitch_col] if pitch_col in df.columns else -90
            height = row[height_col] if height_col in df.columns else 0
            
            meta_dict[stem] = {
                'pitch': float(pitch),
                'height': float(height)
            }
        return meta_dict

    except Exception as e:
        print(f"❌ Error loading CSV {csv_path}: {e}")
        return {}

def find_gt_path_bench(pred_path, gt_root_base):
    """Bench 数据集的 GT 查找逻辑"""
    scene_dir_name = pred_path.parent.name # Cleaned_Dataset_Campus
    file_name = pred_path.name # ImageHash.npy
    
    # 构造 GT 路径
    gt_path = Path(gt_root_base) / scene_dir_name / "depth" / file_name
    
    if gt_path.exists():
        return gt_path
    
    # 备用：有的可能叫 _depth.npy
    stem = pred_path.stem
    gt_path_alt = Path(gt_root_base) / scene_dir_name / "depth" / f"{stem}_depth.npy"
    if gt_path_alt.exists():
        return gt_path_alt
        
    return None

def align_scale_shift_torch(pred, target, mask):
    """最小二乘法对齐"""
    safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    t_valid = target[safe_mask]
    p_valid = pred[safe_mask]
    if len(t_valid) < 10: return pred
    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s, t = solution[0].item(), solution[1].item()
    except: s, t = 1.0, 0.0
    if s <= 1e-4 or np.isnan(s): s = 1.0
    return pred * s + t

def compute_metrics(gt, pred, mask):
    """计算指标 (与 Oblique 一致)"""
    gt = gt[mask]
    pred = pred[mask]
    
    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1_10 = (thresh < 1.10).float().mean()
    a1_25 = (thresh < 1.25).float().mean()
    
    # SI-Log
    log_diff = torch.log(pred) - torch.log(gt)
    si_log = torch.sqrt(torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2)

    # N-RMSE
    p_min, p_max = pred.min(), pred.max()
    g_min, g_max = gt.min(), gt.max()
    p_norm = (pred - p_min) / (p_max - p_min + 1e-8)
    g_norm = (gt - g_min) / (g_max - g_min + 1e-8)
    n_rmse = torch.sqrt(torch.mean((p_norm - g_norm) ** 2))

    # Spearman
    p_cpu = pred.detach().cpu().numpy()
    g_cpu = gt.detach().cpu().numpy()
    if len(p_cpu) > 5000:
        idx = np.random.choice(len(p_cpu), 5000, replace=False)
        p_cpu = p_cpu[idx]
        g_cpu = g_cpu[idx]
    
    if np.std(p_cpu) < 1e-6 or np.std(g_cpu) < 1e-6:
        spearman = 0.0
    else:
        spearman, _ = spearmanr(p_cpu, g_cpu)

    # 0:AbsRel, 1:RMSE, 2:a1.10, 3:a1.25, 4:SI-Log, 5:Spearman, 6:N-RMSE
    return torch.tensor([abs_rel, rmse, a1_10, a1_25, si_log, spearman, n_rmse])

def run_bench_evaluation(pred_root, gt_root_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_root = Path(pred_root)
    
    print(f"🚀 开始 Bench 数据集评估")
    print(f"Pred Root: {pred_root}")
    print(f"GT Root:   {gt_root_base}")

    # 1. 预加载所有 CSV 元数据
    full_meta = {}
    print("\n📖 正在加载 CSV 元数据...")
    for folder_name, csv_path in CSV_MAP.items():
        if os.path.exists(csv_path):
            full_meta[folder_name] = load_csv_metadata(csv_path)
            print(f"  - {folder_name}: 加载了 {len(full_meta[folder_name])} 条记录")
        else:
            print(f"  ⚠️ 找不到 CSV: {csv_path}")
            full_meta[folder_name] = {}

    # 2. 扫描文件
    pred_files = sorted(list(pred_root.rglob("*.npy")))
    print(f"\n✅ 找到 {len(pred_files)} 个预测文件，开始匹配...")

    # 统计容器
    overall_stats = []
    scene_stats = defaultdict(list)
    pitch_stats = defaultdict(list)  # 按角度分 (-90, -60 等)
    height_stats = defaultdict(list) # 按高度分 (0-50m, 50-100m 等)
    
    # 🔥 [新增]: 用于保存逐图详细数据的列表
    per_image_records = [] 

    pbar = tqdm(pred_files)
    valid_count = 0
    
    for pred_path in pbar:
        scene_name = pred_path.parent.name # Cleaned_Dataset_Campus
        stem = pred_path.stem

        # --- 步骤 A: 找 GT ---
        gt_path = find_gt_path_bench(pred_path, gt_root_base)
        if gt_path is None: continue

        # --- 步骤 B: 找 CSV 元数据 ---
        meta = full_meta.get(scene_name, {}).get(stem, None)
        pitch = meta['pitch'] if meta else -90.0
        height = meta['height'] if meta else 0.0

        # --- 步骤 C: 读取与计算 ---
        try:
            pred_np = np.load(pred_path)
            gt_np = np.load(gt_path)
        except: continue
        
        pred = torch.from_numpy(pred_np).to(device).squeeze()
        gt = torch.from_numpy(gt_np).to(device).squeeze()
        
        if pred.shape != gt.shape:
             pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()
        
        mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt)
        if mask.sum() < 10: continue

        if USE_MEDIAN_SCALING:
            pred = align_scale_shift_torch(pred, gt, mask)
        pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

        metrics = compute_metrics(gt, pred, mask) # Tensor
        metrics_np = metrics.cpu().numpy()

        # --- 步骤 D: 归档统计 ---
        overall_stats.append(metrics_np)
        scene_stats[scene_name].append(metrics_np)
        
        # Pitch 分组
        if pitch <= -85: p_key = "Nadir (-90)"
        elif pitch <= -60: p_key = "Oblique (-60 to -85)"
        elif pitch <= -45: p_key = "High Oblique (-45 to -60)"
        else: p_key = "Horizontal (>-45)"
        pitch_stats[p_key].append(metrics_np)

        # Height 分组
        if height < 60: h_key = "Low (<60m)"
        elif height < 120: h_key = "Mid (60-120m)"
        else: h_key = "High (>120m)"
        height_stats[h_key].append(metrics_np)
        
        # 🔥 [新增]: 将单张图的详细信息加入记录列表
        per_image_records.append({
            "Scene": scene_name,
            "Filename": stem,
            "Pitch": pitch,
            "Pitch_Group": p_key,
            "Height": height,
            "Height_Group": h_key,
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

    # ================= 3. 生成总结报告 =================
    
    headers = ["Group/Scene", "N", "AbsRel", "RMSE", "a1.10", "a1.25", "SI-Log", "Spear", "N-RMSE"]
    header_fmt = "{:<28} | {:<4} | {:<7} | {:<7} | {:<6} | {:<6} | {:<6} | {:<6} | {:<6}"
    data_fmt   = "{:<28} | {:<4} | {:<7.4f} | {:<7.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f} | {:<6.3f}"
    
    print("\n" + "="*125)
    print(header_fmt.format(*headers))
    print("-" * 125)

    if overall_stats:
        m = np.mean(overall_stats, axis=0)
        print(data_fmt.format("OVERALL", len(overall_stats), *m))
    print("-" * 125)

    print(">>> By Scene Dataset:")
    for s in sorted(scene_stats.keys()):
        m = np.mean(scene_stats[s], axis=0)
        print(data_fmt.format(s, len(scene_stats[s]), *m))
    print("-" * 125)

    print(">>> By Pitch (Angle):")
    pitch_order = ["Nadir (-90)", "Oblique (-60 to -85)", "High Oblique (-45 to -60)", "Horizontal (>-45)"]
    for p in pitch_order:
        if p in pitch_stats:
            m = np.mean(pitch_stats[p], axis=0)
            print(data_fmt.format(p, len(pitch_stats[p]), *m))
    print("-" * 125)

    print(">>> By Height (Altitude):")
    height_order = ["Low (<60m)", "Mid (60-120m)", "High (>120m)"]
    for h in height_order:
        if h in height_stats:
            m = np.mean(height_stats[h], axis=0)
            print(data_fmt.format(h, len(height_stats[h]), *m))

    # 保存总结报告
    output_file = pred_root / "Eval_Report_Bench.txt"
    with open(output_file, "w") as f:
        f.write(f"Bench Evaluation\nPred: {pred_root}\nGT: {gt_root_base}\n\n")
        f.write(header_fmt.format(*headers) + "\n")
        f.write("-" * 125 + "\n")
        
        if overall_stats:
            m = np.mean(overall_stats, axis=0)
            f.write(data_fmt.format("OVERALL", len(overall_stats), *m) + "\n")
        f.write("-" * 125 + "\n")
        
        for name, stats_dict in [("By Scene", scene_stats), ("By Pitch", pitch_stats), ("By Height", height_stats)]:
            f.write(f">>> {name}:\n")
            for k in sorted(stats_dict.keys()):
                m = np.mean(stats_dict[k], axis=0)
                f.write(data_fmt.format(k, len(stats_dict[k]), *m) + "\n")
            f.write("-" * 125 + "\n")

    print(f"\n📝 总结报告已保存: {output_file}")

    # 🔥 [新增]: 4. 导出详细逐图数据到 CSV
    if per_image_records:
        df_detailed = pd.DataFrame(per_image_records)
        detailed_output_file = pred_root / "Eval_Report_Bench_Detailed.csv"
        # 存成 CSV 格式，数值保留4位小数，Excel可以直接双击打开
        df_detailed.to_csv(detailed_output_file, index=False, float_format="%.4f", encoding='utf-8-sig')
        print(f"📊 逐图详细数据已保存: {detailed_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径根据你的描述设置
    parser.add_argument("--pred", default="/data1/szq/Inference_Results_final_Extracted/Val/Bench", help="Bench 预测目录 (扁平化后)")
    parser.add_argument("--gt", default="/data1/szq/Val/Bench", help="Bench 原始数据集根目录")
    args = parser.parse_args()
    
    if not os.path.exists(args.pred):
        print(f"❌ 错误: 预测目录不存在 {args.pred}")
    else:
        run_bench_evaluation(args.pred, args.gt)
