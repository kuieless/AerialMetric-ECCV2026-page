

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
import json
import matplotlib
matplotlib.use('Agg') # 服务器端绘图必备
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 尝试导入 seaborn 用于高级绘图
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Warning: Seaborn not found. Heatmaps will not be generated. (pip install seaborn)")

# ================= 1. 配置区域 =================
BASE_GT_DIR = "/home/szq/moge2/DJI-self2-final"
BASE_PRED_DIR = "/data1/szq/self1/extracted"

USE_ALIGNMENT = False  # False = 原始绝对深度评估

TASKS = [
    {"name": "campus", "pred_root": os.path.join(BASE_PRED_DIR, "campus"), "gt_root": os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Campus/depth"), "csv_path": os.path.join(BASE_GT_DIR, "final_dataset_campus.csv")},
    {"name": "factory", "pred_root": os.path.join(BASE_PRED_DIR, "factory"), "gt_root": os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Factory/depth"), "csv_path": os.path.join(BASE_GT_DIR, "final_dataset_factory.csv")},
    {"name": "farm", "pred_root": os.path.join(BASE_PRED_DIR, "farm"), "gt_root": os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Farm/depth"), "csv_path": os.path.join(BASE_GT_DIR, "final_dataset_farm.csv")},
    {"name": "gress", "pred_root": os.path.join(BASE_PRED_DIR, "gress"), "gt_root": os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Gress/depth"), "csv_path": os.path.join(BASE_GT_DIR, "final_dataset_grass.csv")},
]

MIN_EVAL_DEPTH, MAX_EVAL_DEPTH = 1e-3, 400
BATCH_SIZE, NUM_WORKERS = 1, 8
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. [关键修改] 分段逻辑 =================

def get_pitch_key(p):
    """
    按照 -90, -75, -60, -45 四类进行严格分段。
    使用中点作为分界线：
    -90: < -82.5
    -75: [-82.5, -67.5)
    -60: [-67.5, -52.5)
    -45: >= -52.5
    """
    if p < -82.5: return "1. -90°"
    if p < -67.5: return "2. -75°"
    if p < -52.5: return "3. -60°"
    return "4. -45°"

def get_alt_key(h):
    # 按照高度排序
    if h < 50: return "1. Low (<50m)"
    if h < 100: return "2. Mid (50-100m)"
    if h < 200: return "3. High (100-200m)"
    return "4. V-High (>200m)"

def get_fov_key(f):
    val = int(round(f))
    return f"~{val}°"

# ================= 3. 核心计算 (单图) =================

def compute_metrics_detailed(gt, pred, valid_mask):
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    
    # 像素级剔除 NaN/Inf
    pixel_mask = valid_mask & torch.isfinite(gt) & torch.isfinite(pred) & (gt_c > 1e-7)
    mask_f = pixel_mask.float()
    v_pix = mask_f.sum()
    
    if v_pix < 10: return None # 无效图返回 None
        
    diff = gt_c - pred_c
    abs_rel = (torch.abs(diff) / gt_c * mask_f).sum() / v_pix
    
    sq_rel = ((diff ** 2) / gt_c * mask_f).sum() / v_pix
    rmse = torch.sqrt((diff ** 2 * mask_f).sum() / v_pix)
    
    thresh = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    a1 = ((thresh < 1.25) * mask_f).sum() / v_pix
    
    return {
        "AbsRel": abs_rel.item(),
        "RMSE": rmse.item(),
        "a1": a1.item(),
        "SqRel": sq_rel.item()
    }

def align_scale_shift_torch(pred, target, mask):
    safe_mask = mask & torch.isfinite(pred) & torch.isfinite(target)
    if safe_mask.sum() < 10: return pred
    t_valid, p_valid = target[safe_mask], pred[safe_mask]
    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s, t = solution[0].item(), solution[1].item()
        if np.isnan(s) or np.isinf(s) or abs(s) > 1e6: raise ValueError()
        return pred * s + t
    except:
        med_t, med_p = torch.median(t_valid), torch.median(p_valid)
        s = (med_t / med_p).item() if med_p > 1e-7 else 1.0
        return pred * s

# ================= 4. 数据集 =================

class RobustDataset(Dataset):
    def __init__(self, csv_path, pred_root, gt_root):
        self.data = []
        try:
            df = pd.read_csv(csv_path, header=None)
            for idx, row in df.iterrows():
                if idx == 0 and not str(row[1]).lower().endswith(".jpg"): continue
                scene = str(row[0]).strip()
                name1, name2 = str(row[1]).strip(), str(row[2]).strip()
                stem1, stem2 = os.path.splitext(name1)[0], os.path.splitext(name2)[0]
                
                p_file = self._find(Path(pred_root), scene, [stem1, stem2], False)
                g_file = self._find(Path(gt_root), scene, [stem1, stem2], True)
                
                if p_file and g_file:
                    self.data.append({
                        "pred": p_file, "gt": g_file,
                        "gt_fov": float(row[5]), "gt_alt": float(row[6]), "gt_pitch": float(row[7])
                    })
        except: pass

    def _find(self, root, scene, stems, is_gt):
        for s in stems:
            cands = [f"{s}.npy"] + (["depth.npy", "gt.npy"] if is_gt else ["depth.npy"])
            for d in [root/scene, root, root/scene/stems[0], root/stems[0]]:
                if d.exists():
                    for c in cands:
                        if (d/c).exists(): return d/c
        return None

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        try:
            return {"pred": torch.from_numpy(np.load(d["pred"])), "gt": torch.from_numpy(np.load(d["gt"])), 
                    "gt_alt": d["gt_alt"], "gt_fov": d["gt_fov"], "gt_pitch": d["gt_pitch"]}
        except: return None

# ================= 5. 绘图函数 (a1指标) =================

def draw_heatmaps(df, output_dir, scene_name):
    if not HAS_SEABORN or df.empty: return

    fov_types = sorted(df['FOV'].unique())
    
    for fov in fov_types:
        df_sub = df[df['FOV'] == fov]
        
        # 这里的 Pitch 已经是修改后的 "1. -90°", "2. -75°"...
        pivot = df_sub.pivot_table(index='Pitch', columns='Altitude', values='a1', aggfunc='mean')
        
        pivot = pivot.sort_index(ascending=True) 
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        if pivot.empty: continue

        plt.figure(figsize=(10, 8))
        # 
        # 热力图展示 a1 精度分布：X轴为高度，Y轴为自定义的 Pitch 类别
        ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", 
                         vmin=0.0, vmax=1.0, 
                         linewidths=.5, cbar_kws={'label': 'a1 Accuracy (Higher is Better)'})
        
        plt.title(f"Scene: {scene_name} | FOV: {fov}\n(a1 Accuracy Distribution)", fontsize=14)
        plt.ylabel("Pitch Angle", fontsize=12)
        plt.xlabel("Altitude", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"Heatmap_a1_FOV_{fov.replace('~', '').replace('°', '')}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    🎨 Heatmap (a1) saved: {os.path.basename(save_path)}")

# ================= 6. 单场景处理 =================

def run_scene_eval(task):
    print(f"\n>>> Analyzing: [{task['name']}]")
    ds = RobustDataset(task['csv_path'], task['pred_root'], task['gt_root'])
    if len(ds) == 0: return None
    dl = DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS, collate_fn=lambda b: [x for x in b if x is not None])
    
    records = []

    for batch in tqdm(dl, leave=False):
        if not batch: continue
        item = batch[0]
        p, g = item['pred'].to(device).squeeze(), item['gt'].to(device).squeeze()
        if p.shape != g.shape: p = F.interpolate(p[None,None], size=g.shape, mode='bilinear').squeeze()
        mask = (g > MIN_EVAL_DEPTH) & (g < MAX_EVAL_DEPTH) & (p > 0)
        
        if USE_ALIGNMENT: p = align_scale_shift_torch(p, g, mask)
        
        metrics = compute_metrics_detailed(g, p, mask)
        if metrics is None: continue 
        
        k_pitch = get_pitch_key(item['gt_pitch'])
        k_alt = get_alt_key(item['gt_alt'])
        k_fov = get_fov_key(item['gt_fov'])
        
        records.append({
            "Pitch": k_pitch,
            "Altitude": k_alt,
            "FOV": k_fov,
            "AbsRel": metrics['AbsRel'],
            "RMSE": metrics['RMSE'],
            "a1": metrics['a1']
        })

    if not records: return None
    
    df = pd.DataFrame(records)
    
    # === 生成报告 ===
    report = []
    report.append(f"SCENE ANALYSIS: {task['name']}")
    report.append(f"Config: Align={USE_ALIGNMENT} | Total Images: {len(df)}")
    report.append("="*110)
    
    header = "{:<55} | {:>5} | {:>8} | {:>8} | {:>8}".format("Condition (Pitch | Alt | FOV)", "Count", "AbsRel", "RMSE", "a1")
    sep = "-" * len(header)
    report += [header, sep]
    
    grouped = df.groupby(['FOV', 'Pitch', 'Altitude'])
    sorted_groups = sorted(grouped.groups.keys())
    
    for key in sorted_groups:
        fov, pitch, alt = key
        group_df = grouped.get_group(key)
        
        count = len(group_df)
        m_abs = group_df['AbsRel'].mean()
        m_rmse = group_df['RMSE'].mean()
        m_a1 = group_df['a1'].mean()
        
        display_key = f"{fov}  |  {pitch}  |  {alt}"
        line = "{:<55} | {:>5} | {:>8.4f} | {:>8.2f} | {:>8.3f}".format(
            display_key, count, m_abs, m_rmse, m_a1)
        report.append(line)
        
    report_text = "\n".join(report)
    with open(os.path.join(task['pred_root'], "Scene_FineGrained_Report.txt"), "w") as f:
        f.write(report_text)
        
    print(f"    Processing heatmaps for {task['name']}...")
    draw_heatmaps(df, task['pred_root'], task['name'])
    
    return df['a1'].mean()

# ================= 6. 主程序 =================

if __name__ == "__main__":
    print(f"🚀 Starting Visual Evaluation (Metric: a1) with Custom Pitch Bins")
    for task in TASKS:
        run_scene_eval(task)
    print(f"\n✅ All Reports and Heatmaps (a1) saved in respective folders.")