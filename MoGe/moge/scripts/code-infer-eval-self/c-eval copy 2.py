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
from scipy.stats import spearmanr

# ================= 1. 配置区域 =================
BASE_GT_DIR = "/home/szq/moge2/DJI-self2-final"
BASE_PRED_DIR = "/data1/szq/self1/extracted"

USE_ALIGNMENT = False  # 是否对齐

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

# ================= 2. 几何计算辅助 =================
# 
# 上图展示了无人机相机俯仰角（Pitch）与视场角（FOV）如何共同决定地面深度分布。

def get_pitch_key(p):
    if p < -75: return "-90°~-75°"
    if p < -60: return "-75°~-60°"
    if p < -45: return "-60°~-45°"
    return "-45°~0°"

# ================= 3. 核心计算函数 =================

def compute_metrics_detailed(gt, pred, valid_mask):
    """计算单张图的 12 项累加指标"""
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    pixel_mask = valid_mask & torch.isfinite(gt) & torch.isfinite(pred) & (gt_c > 1e-7)
    mask_f = pixel_mask.float()
    v_pix = mask_f.sum()
    if v_pix < 10: return torch.zeros(12, device=gt.device)
        
    diff = gt_c - pred_c
    abs_rel = (torch.abs(diff) / gt_c * mask_f).sum()
    sq_rel = ((diff ** 2) / gt_c * mask_f).sum()
    rmse_sq = (diff ** 2 * mask_f).sum()
    log_diff = torch.nan_to_num(torch.log(gt_c) - torch.log(pred_c), 0.0)
    rmse_log_sq = (log_diff ** 2 * mask_f).sum()
    thresh = torch.nan_to_num(torch.maximum((gt_c / pred_c), (pred_c / gt_c)), posinf=999.0)
    a1, a2, a3 = [(thresh < (1.25**i) * mask_f).sum() for i in range(1, 4)]
    
    p_np, g_np = pred_c[pixel_mask].cpu().numpy(), gt_c[pixel_mask].cpu().numpy()
    si_log = np.sqrt(np.mean((np.log(g_np) - np.log(p_np))**2) - np.mean(np.log(g_np) - np.log(p_np))**2 + 1e-10)
    try:
        spear = spearmanr(p_np, g_np)[0]
        spear = 0.0 if np.isnan(spear) else spear
    except: spear = 0.0
    n_rmse = np.sqrt(np.mean(((p_np - p_np.min())/(p_np.max() - p_np.min() + 1e-8) - (g_np - g_np.min())/(g_np.max() - g_np.min() + 1e-8))**2))
    
    return torch.tensor([abs_rel, sq_rel, rmse_sq, rmse_log_sq, a1, a2, a3, v_pix, si_log, spear, n_rmse, 1.0], device=gt.device)

def compute_final_mean(sums):
    pix, img = sums[7], sums[11]
    if pix <= 0: return np.zeros(10)
    m_pix = np.array([sums[0]/pix, sums[1]/pix, np.sqrt(sums[2]/pix), np.sqrt(sums[3]/pix), sums[4]/pix, sums[5]/pix, sums[6]/pix])
    m_img = np.array([sums[8]/img, sums[9]/img, sums[10]/img]) if img > 0 else np.zeros(3)
    return np.concatenate([m_pix, m_img])

# ================= 4. 增强版数据加载 =================

class RobustDataset(Dataset):
    def __init__(self, csv_path, pred_root, gt_root):
        self.data = []
        df = pd.read_csv(csv_path, header=None)
        for idx, row in df.iterrows():
            if idx == 0 and not str(row[1]).lower().endswith(".jpg"): continue
            scene, name1, name2 = str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip()
            stem1, stem2 = os.path.splitext(name1)[0], os.path.splitext(name2)[0]
            
            # 搜索文件
            p_file = self._find(Path(pred_root), scene, [stem1, stem2], False)
            g_file = self._find(Path(gt_root), scene, [stem1, stem2], True)
            
            if p_file and g_file:
                # 寻找 JSON 信息 (FOV, Scale)
                json_path = p_file.parent / f"{os.path.splitext(p_file.name)[0]}.json"
                if not json_path.exists(): json_path = p_file.parent / "fov.json"
                
                self.data.append({
                    "pred": p_file, "gt": g_file, "json": json_path,
                    "gt_fov": float(row[5]), "gt_alt": float(row[6]), "gt_pitch": float(row[7])
                })

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
            pred = np.load(d["pred"])
            gt = np.load(d["gt"])
            # 读取预测的 FOV 和 Scale
            pred_fov, pred_scale = np.nan, 1.0
            if d["json"].exists():
                with open(d["json"], 'r') as f:
                    meta = json.load(f)
                    pred_fov = meta.get("fov_x", np.nan)
                    pred_scale = meta.get("metric_scale", 1.0)
            
            return {"pred": torch.from_numpy(pred), "gt": torch.from_numpy(gt), 
                    "gt_alt": d["gt_alt"], "gt_fov": d["gt_fov"], "gt_pitch": d["gt_pitch"],
                    "pred_fov": pred_fov, "pred_scale": pred_scale}
        except: return None

# ================= 5. 单场景处理逻辑 =================

def run_scene_eval(task):
    print(f"\n>>> Evaluating Scene: [{task['name']}]")
    ds = RobustDataset(task['csv_path'], task['pred_root'], task['gt_root'])
    dl = DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS, collate_fn=lambda b: [x for x in b if x is not None])
    
    # 统计容器
    scene_acc = torch.zeros(12).to(device)
    pitch_stats = {} # {key: acc_tensor}
    geo_data = {"alt_err": [], "fov_err": [], "gt_alt": [], "pred_alt": []}

    for batch in tqdm(dl, leave=False):
        if not batch: continue
        item = batch[0]
        p, g = item['pred'].to(device).squeeze(), item['gt'].to(device).squeeze()
        if p.shape != g.shape: p = F.interpolate(p[None,None], size=g.shape, mode='bilinear').squeeze()
        mask = (g > MIN_EVAL_DEPTH) & (g < MAX_EVAL_DEPTH) & (p > 0)
        
        # 深度指标
        res = compute_metrics_detailed(g, p, mask)
        if res[11] == 0: continue
        
        scene_acc += res
        pk = get_pitch_key(item['gt_pitch'])
        if pk not in pitch_stats: pitch_stats[pk] = torch.zeros(12).to(device)
        pitch_stats[pk] += res
        
        # 几何指标 (高度 & FOV)
        # 高度 = 深度中位数 * 预测尺度 (如果 raw 模式，scale 默认 1.0)
        pred_alt = np.median(p[mask].cpu().numpy()) * item['pred_scale']
        geo_data["alt_err"].append(abs(pred_alt - item['gt_alt']))
        geo_data["gt_alt"].append(item['gt_alt'])
        geo_data["pred_alt"].append(pred_alt)
        if not np.isnan(item['pred_fov']):
            geo_data["fov_err"].append(abs(item['pred_fov'] - item['gt_fov']))

    # 生成该场景报告
    if scene_acc[11] == 0: return None
    
    report = [f"SCENE REPORT: {task['name']}", "="*90]
    header = "{:<15} | {:>5} | {:>7} | {:>7} | {:>7} | {:>7} || {:>7} | {:>7} |".format("Group", "Img#", "AbsRel", "RMSE", "RMSlg", "a1", "SI-Log", "Spear")
    report += [header, "-"*90]
    
    # 场景总计
    m_ov = compute_final_mean(scene_acc.cpu().numpy())
    def fmt(name, m, c): return "{:<15} | {:>5} | {:>7.4f} | {:>7.3f} | {:>7.3f} | {:>7.3f} || {:>7.3f} | {:>7.3f} |".format(name, int(c), m[0], m[2], m[3], m[4], m[7], m[8])
    report.append(fmt("OVERALL", m_ov, scene_acc[11].item()))
    
    # 分 Pitch 统计
    report.append("-" * 90 + "\n[Pitch Analysis]")
    for pk in sorted(pitch_stats.keys()):
        m_p = compute_final_mean(pitch_stats[pk].cpu().numpy())
        report.append(fmt(pk, m_p, pitch_stats[pk][11].item()))
        
    # 高度 & FOV 统计
    report.append("-" * 90 + "\n[Geometry Accuracy]")
    avg_alt_err = np.mean(geo_data["alt_err"])
    mape_alt = (np.array(geo_data["alt_err"]) / np.array(geo_data["gt_alt"])).mean() * 100
    report.append(f"Height MAE:   {avg_alt_err:.4f} m")
    report.append(f"Height MAPE:  {mape_alt:.2f} %")
    if geo_data["fov_err"]:
        report.append(f"FOV MAE:      {np.mean(geo_data['fov_err']):.4f}°")
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(os.path.join(task['pred_root'], "Scene_Detailed_Report.txt"), "w") as f:
        f.write(report_text)
    
    return {"name": task['name'], "alt_mae": avg_alt_err, "alt_mape": mape_alt, "metrics": m_ov}

# ================= 6. 主程序 =================

if __name__ == "__main__":
    print(f"🚀 Starting Scene-Wise Multi-Metric Evaluation")
    summary = []
    for task in TASKS:
        res = run_scene_eval(task)
        if res: summary.append(res)
    
    # 最终汇总大表
    if summary:
        print("\n" + "="*80 + "\nFINAL SUMMARY ACROSS ALL SCENES\n" + "="*80)
        print("{:<10} | {:>10} | {:>10} | {:>7} | {:>7}".format("Scene", "Alt MAE", "Alt MAPE", "AbsRel", "a1"))
        print("-" * 80)
        for s in summary:
            print("{:<10} | {:>10.4f} | {:>9.2f}% | {:>7.4f} | {:>7.3f}".format(s['name'], s['alt_mae'], s['alt_mape'], s['metrics'][0], s['metrics'][4]))