
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
from scipy.stats import spearmanr
import argparse

# ================= 配置 =================
parser = argparse.ArgumentParser()
parser.add_argument("--pred_root", type=str, required=True, help="预测结果根目录")
parser.add_argument("--gt_root", type=str, required=True, help="GT 根目录")
parser.add_argument("--csv_path", type=str, required=True, help="CSV 索引文件")
args = parser.parse_args()

PRED_ROOT = args.pred_root
VAL_GT_ROOTS = [args.gt_root]
MASTER_CSV = args.csv_path

MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 400
BATCH_SIZE = 1
NUM_WORKERS = 8

SCENE_CAT_MAP = {}
_RURAL = ["ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output", "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled", "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled", "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled", "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output", "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled"]
_NATURAL = ["lewis-output", "park5", "park13", "park14", "park10", "park0", "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled", "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled", "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled", "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled", "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled", "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"]
_CITY = ["yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"]
_FACTORY = ["BC2", "BC1", "S-output"]

for s in _RURAL: SCENE_CAT_MAP[s] = "Rural"
for s in _NATURAL: SCENE_CAT_MAP[s] = "Natural"
for s in _CITY: SCENE_CAT_MAP[s] = "City"
for s in _FACTORY: SCENE_CAT_MAP[s] = "Factory"

def get_category(scene): return SCENE_CAT_MAP.get(scene.strip(), "Uncategorized")

warnings.filterwarnings("ignore")

# ================= 几何转换工具 =================

def fov_to_focal(fov_deg, image_size):
    return (image_size / 2) / np.tan(np.deg2rad(fov_deg) / 2)

def focal_to_fov(focal_px, image_size):
    return 2 * np.rad2deg(np.arctan(image_size / (2 * focal_px)))

# ================= 核心对齐算法 =================

def sanitize(tensor):
    return torch.nan_to_num(tensor, nan=1.0, posinf=MAX_EVAL_DEPTH, neginf=MIN_EVAL_DEPTH)

def get_scale_median(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) < 1: return 1.0
    ratio = torch.median(g) / (torch.median(p) + 1e-8)
    return sanitize(ratio).item()

def get_scale_lsq(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) < 1: return 1.0
    pp = torch.dot(p, p)
    pg = torch.dot(p, g)
    ratio = pg / (pp + 1e-8)
    return sanitize(ratio).item()

def get_affine_lsq(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if len(p) < 10: return 1.0, 0.0
    A = torch.stack([p, torch.ones_like(p)], dim=1)
    try:
        sol = torch.linalg.lstsq(A, g, driver='gels').solution
        s, t = sol[0], sol[1]
        s = sanitize(s).item()
        t = sanitize(t).item()
        if s < 1e-4: return get_scale_median(pred, gt, mask), 0.0
        return s, t
    except:
        return get_scale_median(pred, gt, mask), 0.0

# ================= 全指标计算 =================

def compute_all_metrics(gt_map, pred_map, mask):
    gt = gt_map[mask]
    pred = pred_map[mask]
    
    gt = torch.clamp(gt, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    pred = torch.clamp(pred, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    
    n_pixels = len(gt)
    if n_pixels == 0: return torch.zeros(12, device=gt.device)

    diff = gt - pred
    abs_rel = (torch.abs(diff) / gt).sum()
    sq_rel = ((diff ** 2) / gt).sum()
    rmse_sq = (diff ** 2).sum() 
    
    log_diff = torch.log(gt) - torch.log(pred)
    rmse_log_sq = (log_diff ** 2).sum()
    
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().sum()
    a2 = (thresh < 1.25 ** 2).float().sum()
    a3 = (thresh < 1.25 ** 3).float().sum()
    
    si_log_mse = torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2
    si_log = torch.sqrt(torch.relu(si_log_mse)) 
    
    p_np = pred.detach().cpu().numpy()
    g_np = gt.detach().cpu().numpy()
    if len(p_np) > 5000:
        idx = np.random.choice(len(p_np), 5000, replace=False)
        p_np, g_np = p_np[idx], g_np[idx]
    
    try:
        spear, _ = spearmanr(p_np, g_np)
        if np.isnan(spear): spear = 0.0
    except: spear = 0.0
    
    p_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    g_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    n_rmse = torch.sqrt(torch.mean((p_norm - g_norm)**2))

    return torch.tensor([
        abs_rel, sq_rel, rmse_sq, rmse_log_sq, a1, a2, a3, float(n_pixels),
        si_log, spear, n_rmse, 1.0
    ], device=gt.device)

# ================= Smart Dataset (已修复嵌套文件夹读取) =================
class SmartDataset(Dataset):
    def __init__(self, csv_path, pred_root, gt_roots):
        self.df = pd.read_csv(csv_path)
        self.val_df = self.df[self.df['Split'] == 'val'].reset_index(drop=True)
        self.pred_root = Path(pred_root)
        self.gt_roots = [Path(p) for p in gt_roots]
        self.debug_count = 0 
        print(f"Dataset Initialized. Val samples: {len(self.val_df)}")

    def __len__(self): return len(self.val_df)
    
    def find_file(self, roots, scene, sub, stems):
        for root in roots:
            base = root / scene / sub
            if not base.exists(): continue
            
            for stem in stems:
                # === 关键修改：增加了 stem/depth.npy 这种嵌套结构的查找 ===
                candidates = [
                    f"{stem}/depth.npy",   # <--- 优先匹配你的新结构：000001/depth.npy
                    f"{stem}.npy",         # 兼容旧结构
                    f"{stem}_depth.npy", 
                    f"{stem}_Depth.npy", 
                    "depth.npy", 
                    "pred.npy"
                ]
                
                # 1. 精确匹配
                for cand in candidates:
                    f = base / cand
                    if f.exists(): return f
                
                # 2. 忽略大小写匹配 (Linux下有时候文件夹是大写，stem是小写，反之亦然)
                try:
                    real_files = os.listdir(base)
                    lower_map = {f.lower(): f for f in real_files}
                    
                    # 针对嵌套文件夹的特殊处理
                    # 如果 base 下有一个文件夹叫 stem (忽略大小写)，我们要进去找 depth.npy
                    if stem.lower() in lower_map:
                         nested_dir = base / lower_map[stem.lower()]
                         if nested_dir.is_dir():
                             if (nested_dir / "depth.npy").exists(): return nested_dir / "depth.npy"
                             if (nested_dir / "depth.NPY").exists(): return nested_dir / "depth.NPY"

                    # 普通文件匹配
                    for cand in candidates:
                        if cand.lower() in lower_map: 
                            return base / lower_map[cand.lower()]
                except: pass
        return None

    def __getitem__(self, idx):
        row = self.val_df.iloc[idx]
        scene = str(row['Scene_Name']).strip()
        
        renamed = str(row.get('Renamed_Image', ''))
        original = str(row.get('Original_Filename', ''))
        stems = []
        if pd.notna(renamed) and renamed != 'nan': stems.append(os.path.splitext(renamed)[0])
        if pd.notna(original) and original != 'nan': stems.append(os.path.splitext(original)[0])
        
        # 查找文件
        # GT: 你的日志显示在 /Scene/depth/xxx.npy，所以这里用 sub="depth"
        gt_path = self.find_file(self.gt_roots, scene, "depth", stems)
        
        # Pred: 你的结构是 /Scene/0001/depth.npy，所以我们把 sub设为 ""，让 find_file 去匹配 0001/depth.npy
        pred_path = self.find_file([self.pred_root], scene, "", stems)
        
        if gt_path is None or pred_path is None: 
            # 仅在找不到时打印一次（避免刷屏），帮助确认是否修复
            if self.debug_count < 3:
                print(f"❌ Still missing sample {idx} ({scene}): GT={gt_path is not None}, Pred={pred_path is not None}")
                self.debug_count += 1
            return None 
            
        try:
            gt = np.load(gt_path).astype(np.float32)
            pred = np.load(pred_path).astype(np.float32)
            
            focal_val = row.get('FocalLength_New', row.get('FocalLength(px)', None))
            if focal_val is None: focal_val = 0.0
            
            meta = {
                'scene': scene,
                'focal_gt': float(focal_val),
                'scale': 1.0,  
                'fov_x_pred': None
            }
            
            # 读取 JSON 元数据
            # 你的 scale.json 在 pred_path 的同一级目录下 (例如 .../BC2/000001/scale.json)
            # 所以 pred_path.parent 是正确的
            if pred_path.parent.is_dir():
                try:
                    # 优先找 scale.json / fov.json
                    json_cands = [pred_path.parent / "scale.json", pred_path.parent / "fov.json"]
                    for json_f in json_cands:
                        if json_f.exists():
                            with open(json_f) as f:
                                d = json.load(f)
                                if "metric_scale" in d or "scale" in d:
                                    meta['scale'] = float(d.get("metric_scale", d.get("scale", 1.0)))
                                if "fov_x" in d:
                                    meta['fov_x_pred'] = float(d['fov_x'])
                except: pass

            return torch.from_numpy(pred), torch.from_numpy(gt), meta
            
        except Exception as e:
            return None

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

# ================= 主程序 =================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SmartDataset(MASTER_CSV, PRED_ROOT, VAL_GT_ROOTS)
    loader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS, collate_fn=collate)
    
    stats = {} 
    scale_ratios = {}
    scene_intrinsics_details = {} 
    
    MODES = ["Raw Metric", "Median Align", "LSQ Scale", "LSQ Affine"]
    
    print("🚀 开始全模式评估 (Nested Structure Fixed)...")
    
    for batch_data in tqdm(loader):
        if batch_data is None: continue
        preds, gts, metas = batch_data
        if not preds: continue
        
        p_raw = preds[0].to(device).squeeze()
        gt = gts[0].to(device).squeeze()
        meta = metas[0]
        
        if p_raw.dim() == 3: p_raw = p_raw.squeeze(-1)
        if gt.dim() == 3: gt = gt.squeeze(-1)
        if p_raw.shape != gt.shape:
             p_raw = F.interpolate(p_raw[None,None], gt.shape, mode='bilinear').squeeze()
             
        mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH)
        if mask.sum() < 100: continue
        p_raw = torch.nan_to_num(p_raw, nan=1.0, posinf=MAX_EVAL_DEPTH)
        
        # 1. Scale Analysis
        pred_scale = meta['scale'] if meta['scale'] > 1e-6 else 1.0
        p_rel = p_raw / pred_scale
        s2 = get_scale_lsq(p_rel, gt, mask)
        ratio = pred_scale / s2 if s2 > 1e-6 else 1.0
        
        cat = get_category(meta['scene'])
        if cat not in scale_ratios: scale_ratios[cat] = []
        if meta['scene'] not in scale_ratios: scale_ratios[meta['scene']] = []
        if np.isfinite(ratio): scale_ratios[meta['scene']].append(ratio)
        
        # 2. Depth Metrics
        p0 = p_raw
        s1 = get_scale_median(p_rel, gt, mask); p1 = p_rel * s1
        p2 = p_rel * s2
        s3, t3 = get_affine_lsq(p_rel, gt, mask); p3 = p_rel * s3 + t3
        
        for mode_idx, p_curr in enumerate([p0, p1, p2, p3]):
            if torch.isnan(p_curr).any() or torch.isinf(p_curr).any():
                p_curr = torch.nan_to_num(p_curr, nan=MIN_EVAL_DEPTH, posinf=MAX_EVAL_DEPTH)
            res = compute_all_metrics(gt, p_curr, mask)
            if torch.isnan(res).any(): continue
            if "Overall" not in stats: stats["Overall"] = {m: torch.zeros(12, device=device) for m in range(4)}
            stats["Overall"][mode_idx] += res
            if cat not in stats: stats[cat] = {m: torch.zeros(12, device=device) for m in range(4)}
            stats[cat][mode_idx] += res

        # === 3. Intrinsics Analysis ===
        H, W = gt.shape 
        if meta['fov_x_pred'] is not None and meta['focal_gt'] is not None and meta['focal_gt'] > 0:
            fov_pred_deg = meta['fov_x_pred']
            focal_gt_px = meta['focal_gt']
            fov_gt_deg = focal_to_fov(focal_gt_px, W)
            focal_pred_px = fov_to_focal(fov_pred_deg, W)
            
            scn = meta['scene']
            if scn not in scene_intrinsics_details:
                scene_intrinsics_details[scn] = {'cat': cat, 'vals': []}
            
            scene_intrinsics_details[scn]['vals'].append({
                'fov_p': fov_pred_deg,
                'fov_g': fov_gt_deg,
                'foc_p': focal_pred_px,
                'foc_g': focal_gt_px
            })

    # ================= 结果输出 =================
    output_path = os.path.join(PRED_ROOT, "Final_Full_Report.txt")
    f_log = open(output_path, "w")
    
    def log(text):
        print(text)
        f_log.write(text + "\n")

    def calc_mean(tensor_sum):
        t = tensor_sum.cpu().numpy()
        pix = t[7]; img = t[11]
        if pix == 0 or img == 0: return np.zeros(9)
        return np.array([t[0]/pix, np.sqrt(t[2]/pix), np.sqrt(t[3]/pix), t[4]/pix, t[5]/pix, t[6]/pix, t[8]/img, t[9]/img, t[10]/img])

    log("\n" + "="*160)
    log(f"{'Category / Mode':<30} | {'Img':>4} | {'AbsRel':>6} | {'RMSE':>6} | {'RMSlg':>6} | {'a1':>6} || {'SI-Log':>6} | {'Spear':>6}")
    log("-" * 160)
    
    cats = ["Overall"] + sorted([k for k in stats.keys() if k != "Overall"])
    for cat in cats:
        log(f">>> {cat}")
        if cat not in stats: continue
        for m_idx in range(4):
            m_data = stats[cat][m_idx]
            m_res = calc_mean(m_data)
            count = int(m_data[11].item())
            log(f"{'  '+MODES[m_idx]:<30} | {count:>4} | {m_res[0]:>6.4f} | {m_res[1]:>6.2f} | {m_res[2]:>6.3f} | {m_res[3]:>6.3f} || {m_res[6]:>6.3f} | {m_res[7]:>6.3f}")
        log("-" * 160)

    # --- 内参报告 ---
    log("\n>>> 内参详细报告 (Intrinsics Detail Report)")
    log("-" * 110)
    log(f"{'Scene Name':<40} | {'Pred FOV':<8} | {'GT FOV':<8} | {'FOV Err':<8} | {'Pred Focal':<10} | {'GT Focal':<10} | {'Cat'}")
    log("-" * 110)
    
    intrinsics_rows = []
    for scn, data in scene_intrinsics_details.items():
        vals = data['vals']
        if len(vals) == 0: continue
        avg_fov_p = np.mean([v['fov_p'] for v in vals])
        avg_fov_g = np.mean([v['fov_g'] for v in vals])
        avg_foc_p = np.mean([v['foc_p'] for v in vals])
        avg_foc_g = np.mean([v['foc_g'] for v in vals])
        err_fov = abs(avg_fov_p - avg_fov_g)
        intrinsics_rows.append((scn, avg_fov_p, avg_fov_g, err_fov, avg_foc_p, avg_foc_g, data['cat']))
    
    intrinsics_rows.sort(key=lambda x: x[3], reverse=True)
    
    for row in intrinsics_rows:
        log(f"{row[0]:<40} | {row[1]:<8.2f} | {row[2]:<8.2f} | {row[3]:<8.2f} | {row[4]:<10.1f} | {row[5]:<10.1f} | {row[6]}")

    # --- 尺度偏差报告 ---
    log("\n>>> 尺度偏差分析 (Scale Drift Analysis)")
    log(f"{'Scene Name':<40} | {'Ratio':<10} | {'Std':<6} | {'Interp'}")
    log("-" * 80)
    
    scene_rows = []
    for scn, ratios in scale_ratios.items():
        if isinstance(ratios, list) and len(ratios) > 0:
            r_arr = np.array(ratios)
            m_r = np.mean(r_arr)
            s_r = np.std(r_arr)
            interp = "✅"
            if m_r < 0.8: interp = "🔻 Small"
            if m_r > 1.2: interp = "🔺 Large"
            scene_rows.append((scn, m_r, s_r, interp))
            
    scene_rows.sort(key=lambda x: abs(x[1]-1.0), reverse=True)
    
    for row in scene_rows:
        log(f"{row[0]:<40} | {row[1]:<10.4f} | {row[2]:<6.2f} | {row[3]}")

    f_log.close()
    print(f"\n✅ 完整报告已保存至: {output_path}")

# '''
# python /home/szq/moge2/MoGe/moge/scripts/c-real-tasks-pitch-altitude-fov-syn-debug-batch-classes4-scale???.py --pred_root /data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k --gt_root  /data1/szq/data/Val  --csv_path /home/szq/moge2/final_merged.csv

# '''