# # import os
# # import argparse
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# # from tqdm import tqdm
# # from datetime import datetime
# # from pathlib import Path
# # import warnings
# # from scipy.stats import spearmanr

# # # ================= 1. 配置与参数 =================
# # parser = argparse.ArgumentParser(description="Step 3: Dual Mode Evaluation (Raw & Aligned)")
# # parser.add_argument("--pred_root", type=str, required=True, help="预测结果目录")
# # parser.add_argument("--gt_root", type=str, required=True, help="GT 真值目录")
# # parser.add_argument("--csv_path", type=str, required=True, help="CSV 索引文件路径")
# # args = parser.parse_args()

# # PRED_ROOT = Path(args.pred_root)
# # GT_ROOT = Path(args.gt_root)
# # CSV_PATH = args.csv_path

# # # 评估参数
# # MIN_EVAL_DEPTH = 1e-3
# # MAX_EVAL_DEPTH = 400
# # BATCH_SIZE = 1 
# # NUM_WORKERS = 8
# # FOCAL_STEP = 400

# # # 深度分段
# # DEPTH_BINS = [(0, 50), (50, 120), (120, 250), (250, 400), (400, 500)]
# # BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]
# # ALL_LABELS = BIN_LABELS + ["Overall"]

# # warnings.filterwarnings("ignore")

# # # ================= 2. 场景类别映射 =================
# # _RURAL_LIST = [
# #     "ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output",
# #     "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled",
# #     "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled",
# #     "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled",
# #     "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output",
# #     "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled"
# # ]
# # _NATURAL_LIST = [
# #     "lewis-output", "park5", "park13", "park14", "park10", "park0",
# #     "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled",
# #     "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled",
# #     "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled",
# #     "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled",
# #     "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled",
# #     "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"
# # ]
# # _CITY_LIST = ["yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"]
# # _FACTORY_LIST = ["BC2", "BC1", "S-output"]

# # SCENE_CAT_MAP = {}
# # for s in _RURAL_LIST: SCENE_CAT_MAP[s] = "Rural"
# # for s in _NATURAL_LIST: SCENE_CAT_MAP[s] = "Natural"
# # for s in _CITY_LIST: SCENE_CAT_MAP[s] = "City"
# # for s in _FACTORY_LIST: SCENE_CAT_MAP[s] = "Factory"

# # def get_scene_category(scene_name):
# #     return SCENE_CAT_MAP.get(scene_name.strip(), "Uncategorized")

# # # ================= 3. 核心计算逻辑 =================

# # def align_scale_shift_torch(pred, target, mask):
# #     """鲁棒的最小二乘法对齐 (Scale & Shift)"""
# #     safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
# #     t_valid = target[safe_mask]
# #     p_valid = pred[safe_mask]
    
# #     if len(t_valid) < 10:
# #         return pred 

# #     ones = torch.ones_like(p_valid)
# #     A = torch.stack([p_valid, ones], dim=1)
    
# #     success = False
# #     try:
# #         solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
# #         s = solution[0].item()
# #         t = solution[1].item()
# #         # 简单的物理约束：Scale 必须为正，且不过分极端
# #         if not (np.isnan(s) or np.isnan(t) or np.isinf(s) or np.isinf(t)) and s > 1e-4:
# #             success = True
# #     except: pass
        
# #     if not success:
# #         # 回退到中位数对齐
# #         med_t = torch.median(t_valid)
# #         med_p = torch.median(p_valid)
# #         if med_p > 1e-6:
# #             s = (med_t / med_p).item()
# #             t = 0.0
# #         else:
# #             s, t = 1.0, 0.0

# #     return pred * s + t

# # def compute_errors_torch_bins(gt, pred, valid_mask):
# #     """计算所有指标"""
# #     if pred.dim() == 3 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
# #     if gt.dim() == 3 and gt.shape[-1] == 1: gt = gt.squeeze(-1)
    
# #     gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
# #     pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
# #     rmse_map = (gt - pred) ** 2
# #     rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
# #     abs_rel_map = torch.abs(gt - pred) / gt_c
# #     sq_rel_map = ((gt - pred) ** 2) / gt_c
# #     thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
# #     all_ranges = DEPTH_BINS + [("Overall", "Overall")]
# #     batch_results = []
    
# #     for (b_min, b_max) in all_ranges:
# #         if b_min == "Overall":
# #             current_mask = valid_mask
# #         else:
# #             current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
            
# #         mask_f = current_mask.float()
# #         valid_pixel_count = mask_f.sum(dim=[1, 2])
        
# #         a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
# #         a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
# #         a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
# #         rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
# #         rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
# #         abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
# #         sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])

# #         si_log_list = []
# #         spearman_list = []
# #         norm_rmse_list = []
        
# #         for b in range(gt.shape[0]):
# #             m = current_mask[b]
# #             if m.sum() < 10:
# #                 si_log_list.append(0.0); spearman_list.append(0.0); norm_rmse_list.append(0.0)
# #                 continue
                
# #             g_valid = gt_c[b][m]
# #             p_valid = pred_c[b][m]

# #             log_diff = torch.log(g_valid) - torch.log(p_valid)
# #             si_log_val = torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2 + 1e-8)
# #             si_log_list.append(si_log_val.item())
            
# #             p_np = p_valid.detach().cpu().numpy()
# #             g_np = g_valid.detach().cpu().numpy()
# #             if len(p_np) > 5000:
# #                 idx = np.random.choice(len(p_np), 5000, replace=False)
# #                 p_np, g_np = p_np[idx], g_np[idx]
            
# #             if np.std(p_np) < 1e-6 or np.std(g_np) < 1e-6:
# #                 spearman_val = 0.0
# #             else:
# #                 try:
# #                     s_val, _ = spearmanr(p_np, g_np)
# #                     spearman_val = 0.0 if np.isnan(s_val) else s_val
# #                 except: spearman_val = 0.0
# #             spearman_list.append(spearman_val)
            
# #             p_norm = (p_valid - p_valid.min()) / (p_valid.max() - p_valid.min() + 1e-8)
# #             g_norm = (g_valid - g_valid.min()) / (g_valid.max() - g_valid.min() + 1e-8)
# #             norm_rmse_val = torch.sqrt(torch.mean((p_norm - g_norm)**2))
# #             norm_rmse_list.append(norm_rmse_val.item())

# #         si_log_t = torch.tensor(si_log_list, device=gt.device)
# #         spearman_t = torch.tensor(spearman_list, device=gt.device)
# #         norm_rmse_t = torch.tensor(norm_rmse_list, device=gt.device)
# #         valid_img_count = (valid_pixel_count > 10).float()

# #         bin_res = torch.stack([
# #             abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count,
# #             si_log_t, spearman_t, norm_rmse_t, valid_img_count
# #         ], dim=1)
# #         batch_results.append(bin_res)
        
# #     return torch.stack(batch_results, dim=1)

# # def compute_metrics_from_sums(sums):
# #     total_valid_pixels = sums[7]
# #     total_valid_imgs = sums[11]
    
# #     if total_valid_pixels <= 0: return np.zeros(11)
    
# #     pixel_metrics = np.array([
# #         sums[0]/total_valid_pixels, sums[1]/total_valid_pixels,
# #         np.sqrt(sums[2]/total_valid_pixels), np.sqrt(sums[3]/total_valid_pixels),
# #         sums[4]/total_valid_pixels, sums[5]/total_valid_pixels, sums[6]/total_valid_pixels
# #     ])
    
# #     if total_valid_imgs > 0:
# #         img_metrics = np.array([sums[8]/total_valid_imgs, sums[9]/total_valid_imgs, sums[10]/total_valid_imgs])
# #     else:
# #         img_metrics = np.zeros(3)
        
# #     return np.concatenate([pixel_metrics, img_metrics])

# # # ================= 4. 数据加载 (Robust) =================

# # class RobustDataset(Dataset):
# #     def __init__(self, csv_path, pred_root, gt_root):
# #         self.pred_root = Path(pred_root)
# #         self.gt_root = Path(gt_root)
# #         self.data = []
        
# #         print(f"Loading CSV: {csv_path}...")
# #         try:
# #             df = pd.read_csv(csv_path, header=None)
# #             print(f"CSV Loaded. Rows: {len(df)}")
# #             success = 0
# #             for idx, row in df.iterrows():
# #                 if idx == 0 and not str(row[1]).lower().endswith(".jpg"): continue
# #                 scene = str(row[0]).strip()
# #                 name1 = str(row[1]).strip()
# #                 name2 = str(row[2]).strip()
# #                 fov = float(row[5]) if pd.notna(row[5]) else 60.0
# #                 pitch = float(row[7]) if pd.notna(row[7]) else 0.0
                
# #                 stem1 = os.path.splitext(name1)[0]
# #                 stem2 = os.path.splitext(name2)[0]
                
# #                 pred_path = self._find_file(self.pred_root, scene, [stem1, stem2], is_gt=False)
# #                 gt_path = self._find_file(self.gt_root, scene, [stem1, stem2], is_gt=True)
                
# #                 if pred_path and gt_path:
# #                     W_est = 4000
# #                     focal_est = (W_est / 2) / np.tan(np.deg2rad(fov) / 2)
# #                     self.data.append({
# #                         "scene": scene, "pred_path": pred_path, "gt_path": gt_path,
# #                         "pitch": pitch, "focal": focal_est
# #                     })
# #                     success += 1
# #             print(f"✅ Valid Samples: {success}")
# #         except Exception as e: print(f"Error parsing CSV: {e}")

# #     def _find_file(self, root, scene, stems, is_gt=False):
# #         for stem in stems:
# #             candidates = [f"{stem}.npy"]
# #             if is_gt: candidates += ["depth.npy", "gt.npy"]
# #             else: candidates += ["depth.npy"]
# #             search_dirs = [root / scene, root, root / scene / stem, root / stem]
# #             for d in search_dirs:
# #                 if not d.exists(): continue
# #                 for cand in candidates:
# #                     f = d / cand
# #                     if f.exists(): return f
# #         return None

# #     def __len__(self): return len(self.data)
# #     def __getitem__(self, idx):
# #         item = self.data[idx]
# #         try:
# #             pred = np.load(item["pred_path"])
# #             gt = np.load(item["gt_path"])
# #             meta = {'scene': item['scene'], 'pitch': item['pitch'], 'focal': item['focal']}
# #             return torch.from_numpy(pred), torch.from_numpy(gt), meta
# #         except: return None

# # def collate_fn(batch):
# #     batch = [b for b in batch if b is not None]
# #     if not batch: return None
# #     return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

# # # ================= 5. 统计类 =================

# # class StatsTracker:
# #     def __init__(self, device):
# #         self.device = device
# #         self.acc_overall = torch.zeros(len(ALL_LABELS), 12).to(device)
# #         self.acc_scene = {}
# #         self.acc_cat = {}
# #         self.acc_pitch = {}
# #         self.acc_focal = {}
        
# #         self.count_scene = {}
# #         self.count_cat = {}
# #         self.count_pitch = {}
# #         self.count_focal = {}
# #         self.total_imgs = 0

# #     def update(self, batch_res, meta):
# #         self.total_imgs += 1
# #         self.acc_overall += batch_res
        
# #         # Helper update function
# #         def _up(d_acc, d_cnt, key):
# #             if key not in d_acc:
# #                 d_acc[key] = torch.zeros_like(batch_res)
# #                 d_cnt[key] = {'img': 0, 'scn': set()}
# #             d_acc[key] += batch_res
# #             d_cnt[key]['img'] += 1
# #             d_cnt[key]['scn'].add(meta['scene'])

# #         # Scene
# #         _up(self.acc_scene, self.count_scene, meta['scene'])
        
# #         # Category
# #         cat = get_scene_category(meta['scene'])
# #         _up(self.acc_cat, self.count_cat, cat)
        
# #         # Pitch
# #         p = meta['pitch']
# #         if p < -75: pk = "-90to-75"
# #         elif p < -60: pk = "-75to-60"
# #         elif p < -45: pk = "-60to-45"
# #         else: pk = "-45to-0"
# #         _up(self.acc_pitch, self.count_pitch, pk)
        
# #         # Focal
# #         f = meta['focal']
# #         fk = f"{int(f//FOCAL_STEP)*FOCAL_STEP}-{(int(f//FOCAL_STEP)+1)*FOCAL_STEP}"
# #         _up(self.acc_focal, self.count_focal, fk)

# # # ================= 6. 报告生成 =================

# # def format_line(name, m, count_img=0, count_scn=0):
# #     m = np.nan_to_num(m)
# #     return "{:<20} | {:>4} | {:>4} | {:>6.4f} | {:>6.2f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:>6.3f} | {:>6.3f} | {:>6.3f} |".format(
# #         str(name)[-20:], count_img, count_scn,
# #         m[0], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9]
# #     )

# # def format_header():
# #     return "{:<20} | {:>4} | {:>4} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} || {:>6} | {:>6} | {:>6} |".format(
# #         "Category", "Img#", "Scn#", "AbsRel", "RMSE", "RMSlg", "a1", "a2", "a3", "SI-Log", "Spear", "N-RMSE")

# # def generate_report_text(stats, title):
# #     if stats.total_imgs == 0: return "No Data"
    
# #     lines = []
# #     header = format_header()
# #     sep = "-" * len(header)
# #     lines += [f"=== {title} ===", f"Date: {datetime.now()}", sep, header, sep]
    
# #     # Overall
# #     ov_res = compute_metrics_from_sums(stats.acc_overall[-1].cpu().numpy())
# #     all_scn = len(stats.count_scene)
# #     lines.append(format_line("OVERALL", ov_res, stats.total_imgs, all_scn))
# #     lines.append(sep)
    
# #     # Category
# #     lines.append(">>> BY CATEGORY")
# #     for k in ["Rural", "Natural", "City", "Factory", "Uncategorized"]:
# #         if k in stats.acc_cat:
# #             res = compute_metrics_from_sums(stats.acc_cat[k][-1].cpu().numpy())
# #             cnt = stats.count_cat[k]
# #             lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
# #     lines.append(sep)
    
# #     # Pitch
# #     lines.append(">>> BY PITCH")
# #     for k in sorted(stats.acc_pitch.keys()):
# #         res = compute_metrics_from_sums(stats.acc_pitch[k][-1].cpu().numpy())
# #         cnt = stats.count_pitch[k]
# #         lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
# #     lines.append(sep)

# #     return "\n".join(lines)

# # # ================= 7. 主程序 =================

# # if __name__ == "__main__":
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# #     dataset = RobustDataset(CSV_PATH, PRED_ROOT, GT_ROOT)
# #     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
# #     # 初始化两个统计器
# #     stats_raw = StatsTracker(device)
# #     stats_aligned = StatsTracker(device)
    
# #     print("🚀 开始双模式评估 (Raw & Aligned)...")
    
# #     for batch in tqdm(loader):
# #         if batch is None: continue
# #         preds, gts, metas = batch
        
# #         for i in range(len(preds)):
# #             pred_raw = preds[i].to(device)
# #             gt = gts[i].to(device)
# #             meta = metas[i]
            
# #             # 尺寸对齐
# #             if pred_raw.shape != gt.shape:
# #                 pred_raw = F.interpolate(pred_raw.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()
            
# #             mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH) & (pred_raw > 0)
# #             if mask.sum() < 100: continue
            
# #             # === Mode 1: Raw ===
# #             pred_raw_c = torch.clamp(pred_raw, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
# #             res_raw = compute_errors_torch_bins(gt.unsqueeze(0), pred_raw_c.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
# #             stats_raw.update(res_raw, meta)
            
# #             # === Mode 2: Aligned (Least Squares) ===
# #             pred_aligned = align_scale_shift_torch(pred_raw, gt, mask)
# #             pred_aligned_c = torch.clamp(pred_aligned, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
# #             res_aligned = compute_errors_torch_bins(gt.unsqueeze(0), pred_aligned_c.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
# #             stats_aligned.update(res_aligned, meta)

# #     # 生成报告
# #     report_raw = generate_report_text(stats_raw, "RAW EVALUATION (No Alignment)")
# #     report_aligned = generate_report_text(stats_aligned, "ALIGNED EVALUATION (Least Squares)")
    
# #     full_report = report_raw + "\n\n" + "="*130 + "\n\n" + report_aligned
    
# #     print(full_report)
    
# #     out_path = PRED_ROOT / "Final_Dual_Report.txt"
# #     with open(out_path, "w") as f:
# #         f.write(full_report)
# #     print(f"\n✅ 双模式报告已保存: {out_path}")



# import os
# import argparse
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from datetime import datetime
# from pathlib import Path
# import warnings
# from scipy.stats import spearmanr
# import sys

# # ================= 1. 任务配置区域 (在这里修改路径) =================

# # CSV 和 GT 的基础目录
# BASE_GT_DIR = "/home/szq/moge2/DJI-self2-final"
# BASE_PRED_DIR = "/data1/szq/self/extracted"

# # 定义 4 个任务
# TASKS = [
#     {
#         "name": "campus",
#         "pred_root": os.path.join(BASE_PRED_DIR, "campus"),
#         "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Campus/depth"),
#         "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_campus.csv")
#     },
#     {
#         "name": "factory",
#         "pred_root": os.path.join(BASE_PRED_DIR, "factory"),
#         "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Factory/depth"),
#         "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_factory.csv")
#     },
#     {
#         "name": "farm",
#         "pred_root": os.path.join(BASE_PRED_DIR, "farm"),
#         "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Farm/depth"),
#         "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_farm.csv")
#     },
#     {
#         "name": "gress",
#         "pred_root": os.path.join(BASE_PRED_DIR, "gress"),
#         "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Gress/depth"),
#         "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_grass.csv") # 注意这里可能是 grass.csv
#     },
# ]

# # 通用参数
# MIN_EVAL_DEPTH = 1e-3
# MAX_EVAL_DEPTH = 400
# BATCH_SIZE = 1 
# NUM_WORKERS = 8
# FOCAL_STEP = 400
# DEPTH_BINS = [(0, 50), (50, 120), (120, 250), (250, 400), (400, 500)]
# ALL_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS] + ["Overall"]

# warnings.filterwarnings("ignore")

# # ================= 2. 核心计算函数 (复用之前的逻辑) =================

# def align_scale_shift_torch(pred, target, mask):
#     safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
#     t_valid = target[safe_mask]
#     p_valid = pred[safe_mask]
#     if len(t_valid) < 10: return pred 

#     ones = torch.ones_like(p_valid)
#     A = torch.stack([p_valid, ones], dim=1)
    
#     success = False
#     try:
#         solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
#         s, t = solution[0].item(), solution[1].item()
#         if not (np.isnan(s) or np.isinf(s)) and s > 1e-4: success = True
#     except: pass
        
#     if not success:
#         med_t, med_p = torch.median(t_valid), torch.median(p_valid)
#         s = (med_t / med_p).item() if med_p > 1e-6 else 1.0
#         t = 0.0

#     return pred * s + t

# def compute_errors_torch_bins(gt, pred, valid_mask):
#     if pred.dim() == 3: pred = pred.squeeze(-1)
#     if gt.dim() == 3: gt = gt.squeeze(-1)
    
#     gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
#     pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
#     thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
#     all_ranges = DEPTH_BINS + [("Overall", "Overall")]
#     batch_results = []
    
#     for (b_min, b_max) in all_ranges:
#         if b_min == "Overall": current_mask = valid_mask
#         else: current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
        
#         mask_f = current_mask.float()
#         valid_pixel_count = mask_f.sum(dim=[1, 2])
        
#         # 基础指标
#         a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
#         a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
#         a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
#         rmse_s = (((gt - pred) ** 2) * mask_f).sum(dim=[1, 2])
#         rmse_log_s = (((torch.log(gt_c) - torch.log(pred_c)) ** 2) * mask_f).sum(dim=[1, 2])
#         abs_rel_s = ((torch.abs(gt - pred) / gt_c) * mask_f).sum(dim=[1, 2])
#         sq_rel_s = ((((gt - pred) ** 2) / gt_c) * mask_f).sum(dim=[1, 2])

#         # 复杂指标 (SI-Log, Spearman)
#         si_log_list, spearman_list = [], []
#         for b in range(gt.shape[0]):
#             m = current_mask[b]
#             if m.sum() < 10:
#                 si_log_list.append(0.0); spearman_list.append(0.0)
#                 continue
            
#             g_v = gt_c[b][m]
#             p_v = pred_c[b][m]
            
#             # SI-Log
#             log_diff = torch.log(g_v) - torch.log(p_v)
#             si_log_list.append(torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2 + 1e-8).item())
            
#             # Spearman
#             p_np, g_np = p_v.detach().cpu().numpy(), g_v.detach().cpu().numpy()
#             if len(p_np) > 5000:
#                 idx = np.random.choice(len(p_np), 5000, replace=False)
#                 p_np, g_np = p_np[idx], g_np[idx]
#             try:
#                 s_val, _ = spearmanr(p_np, g_np)
#                 spearman_list.append(0.0 if np.isnan(s_val) else s_val)
#             except: spearman_list.append(0.0)

#         # 堆叠结果: 0:AbsRel, 1:SqRel, 2:RMSE, 3:RMSElog, 4:a1, 5:a2, 6:a3, 7:PixCount, 8:SI-Log, 9:Spear, 10:ImgCount
#         res = torch.stack([
#             abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count,
#             torch.tensor(si_log_list, device=gt.device), 
#             torch.tensor(spearman_list, device=gt.device),
#             (valid_pixel_count > 10).float()
#         ], dim=1)
#         batch_results.append(res)
        
#     return torch.stack(batch_results, dim=1)

# def compute_metrics_from_sums(sums):
#     pix = sums[7]; img = sums[10]
#     if pix <= 0: return np.zeros(10)
    
#     # Pixel-based
#     m = np.array([sums[0]/pix, sums[1]/pix, np.sqrt(sums[2]/pix), np.sqrt(sums[3]/pix), sums[4]/pix, sums[5]/pix, sums[6]/pix])
#     # Image-based
#     m2 = np.array([sums[8]/img, sums[9]/img]) if img > 0 else np.zeros(2)
#     return np.concatenate([m, m2])

# # ================= 3. 数据加载与处理 =================

# class RobustDataset(Dataset):
#     def __init__(self, csv_path, pred_root, gt_root):
#         self.pred_root = Path(pred_root)
#         self.gt_root = Path(gt_root)
#         self.data = []
#         try:
#             df = pd.read_csv(csv_path, header=None)
#             for idx, row in df.iterrows():
#                 if idx == 0 and not str(row[1]).lower().endswith(".jpg"): continue
#                 scene = str(row[0]).strip()
#                 name1, name2 = str(row[1]).strip(), str(row[2]).strip()
#                 stem1, stem2 = os.path.splitext(name1)[0], os.path.splitext(name2)[0]
                
#                 pred_path = self._find(self.pred_root, scene, [stem1, stem2], False)
#                 gt_path = self._find(self.gt_root, scene, [stem1, stem2], True)
#                 if pred_path and gt_path:
#                     self.data.append({"pred": pred_path, "gt": gt_path})
#         except: pass

#     def _find(self, root, scene, stems, is_gt):
#         cands = [f"{s}.npy" for s in stems]
#         if is_gt: cands += ["depth.npy", "gt.npy"]
#         else: cands += ["depth.npy"]
#         dirs = [root/scene, root, root/scene/stems[0], root/stems[0]]
#         for d in dirs:
#             if not d.exists(): continue
#             for c in cands:
#                 if (d/c).exists(): return d/c
#         return None

#     def __len__(self): return len(self.data)
#     def __getitem__(self, idx):
#         try:
#             return torch.from_numpy(np.load(self.data[idx]["pred"])), torch.from_numpy(np.load(self.data[idx]["gt"]))
#         except: return None

# def collate_fn(b):
#     b = [x for x in b if x is not None]
#     return (None, None) if not b else ([x[0] for x in b], [x[1] for x in b])

# # ================= 4. 单个场景评估流程 =================

# def run_single_task(task, device):
#     print(f"\n[{task['name']}] Processing...")
#     print(f"  Pred: {task['pred_root']}\n  GT:   {task['gt_root']}")
    
#     if not os.path.exists(task['gt_root']):
#         print(f"  ❌ GT Path not found: {task['gt_root']}")
#         return None
        
#     ds = RobustDataset(task['csv_path'], task['pred_root'], task['gt_root'])
#     if len(ds) == 0:
#         print(f"  ❌ No matching files found.")
#         return None
        
#     dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
#     # 累加器: [Bins, 11]
#     acc_raw = torch.zeros(len(ALL_LABELS), 11).to(device)
#     acc_aligned = torch.zeros(len(ALL_LABELS), 11).to(device)
#     count = 0
    
#     for preds, gts in tqdm(dl, desc=f"  Eval {task['name']}", leave=False):
#         if preds is None: continue
#         for i in range(len(preds)):
#             p, g = preds[i].to(device), gts[i].to(device)
            
#             if p.shape != g.shape:
#                 p = F.interpolate(p[None,None], size=g.shape, mode='bilinear').squeeze()
            
#             mask = (g > MIN_EVAL_DEPTH) & (g < MAX_EVAL_DEPTH) & (p > 0)
#             if mask.sum() < 100: continue
            
#             # Raw
#             res_raw = compute_errors_torch_bins(g[None], p[None], mask[None]).squeeze(0)
#             acc_raw += res_raw
            
#             # Aligned
#             p_align = align_scale_shift_torch(p, g, mask)
#             res_align = compute_errors_torch_bins(g[None], p_align[None], mask[None]).squeeze(0)
#             acc_aligned += res_align
#             count += 1
            
#     # 计算平均
#     ov_idx = -1
#     raw_metrics = compute_metrics_from_sums(acc_raw[ov_idx].cpu().numpy())
#     align_metrics = compute_metrics_from_sums(acc_aligned[ov_idx].cpu().numpy())
    
#     # 保存报告
#     report = []
#     header = "{:<10} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}".format(
#         "Mode", "AbsRel", "RMSE", "RMSlg", "a1", "SI-Log", "Spear", "Count")
#     sep = "-" * len(header)
    
#     report += [f"Scene: {task['name']}", sep, header, sep]
    
#     def fmt_row(name, m, cnt):
#         return "{:<10} | {:>6.4f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6}".format(
#             name, m[0], m[2], m[3], m[4], m[7], m[8], cnt)
            
#     report.append(fmt_row("RAW", raw_metrics, count))
#     report.append(fmt_row("ALIGNED", align_metrics, count))
    
#     out_file = os.path.join(task['pred_root'], "Eval_Report_Summary.txt")
#     with open(out_file, 'w') as f: f.write("\n".join(report))
    
#     return {
#         "name": task['name'], 
#         "count": count,
#         "raw": raw_metrics, 
#         "aligned": align_metrics
#     }

# # ================= 5. 主程序 =================

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🚀 Starting Batch Evaluation ({len(TASKS)} Tasks)")
    
#     final_summaries = []
    
#     for task in TASKS:
#         res = run_single_task(task, device)
#         if res: final_summaries.append(res)
        
#     print("\n" + "="*80)
#     print("🌍 FINAL BATCH SUMMARY")
#     print("="*80)
#     print("{:<10} | {:<8} | {:<20} | {:<20}".format("Scene", "Count", "Raw (AbsRel/RMSE/a1)", "Aligned (AbsRel/RMSE/a1)"))
#     print("-" * 80)
    
#     for s in final_summaries:
#         r = s['raw']; a = s['aligned']
#         r_str = f"{r[0]:.3f} / {r[2]:.2f} / {r[4]:.2f}"
#         a_str = f"{a[0]:.3f} / {a[2]:.2f} / {a[4]:.2f}"
#         print("{:<10} | {:<8} | {:<20} | {:<20}".format(s['name'], s['count'], r_str, a_str))
        
#     print("="*80)
#     print("✅ All done. Detailed reports saved in each pred folder.")
#     '''

# python /home/szq/moge2/MoGe/moge/scripts/code-infer-eval-self/c-eval.py \
#   --pred_root /data1/szq/self/extracted/campus \
#   --csv_path /home/szq/moge2/DJI-self2-final/final_dataset_campus.csv \
#   --output_dir /home/szq/Megadepth/eval_results

# python /home/szq/moge2/MoGe/moge/scripts/code-infer-eval-self/c-eval.py \
#   --pred_root /data1/szq/self/extracted/campus \
#   --gt_root /home/szq/moge2/DJI-self2-final/Cleaned_Dataset_Campus/depth \
#   --csv_path /home/szq/moge2/DJI-self2-final/final_dataset_campus.csv \

  
#     '''
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
from scipy.stats import spearmanr
import argparse

# ================= 1. 任务配置 =================

# 你的路径配置
BASE_GT_DIR = "/home/szq/moge2/DJI-self2-final"
BASE_PRED_DIR = "/data1/szq/self1/extracted"

# 定义 4 个任务
TASKS = [
    {
        "name": "campus",
        "pred_root": os.path.join(BASE_PRED_DIR, "campus"),
        "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Campus/depth"),
        "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_campus.csv")
    },
    {
        "name": "factory",
        "pred_root": os.path.join(BASE_PRED_DIR, "factory"),
        "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Factory/depth"),
        "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_factory.csv")
    },
    {
        "name": "farm",
        "pred_root": os.path.join(BASE_PRED_DIR, "farm"),
        "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Farm/depth"),
        "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_farm.csv")
    },
    {
        "name": "gress",
        "pred_root": os.path.join(BASE_PRED_DIR, "gress"),
        "gt_root":   os.path.join(BASE_GT_DIR, "Cleaned_Dataset_Gress/depth"),
        "csv_path":  os.path.join(BASE_GT_DIR, "final_dataset_grass.csv") 
    },
]

# 评估参数
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 400
USE_ALIGNMENT = False
BATCH_SIZE = 1 
NUM_WORKERS = 8
FOCAL_STEP = 400
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 场景类别映射 =================

_RURAL = ["farm", "ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output"]
_NATURAL = ["gress", "lewis-output", "park5", "park13", "park14", "park10", "park0", "bellus-output", "sceneca-output"]
_CITY = ["campus", "yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"]
_FACTORY = ["factory", "BC2", "BC1", "S-output"]

SCENE_CAT_MAP = {}
for s in _RURAL: SCENE_CAT_MAP[s] = "Rural"
for s in _NATURAL: SCENE_CAT_MAP[s] = "Natural"
for s in _CITY: SCENE_CAT_MAP[s] = "City"
for s in _FACTORY: SCENE_CAT_MAP[s] = "Factory"

def get_scene_category(scene_name):
    s = scene_name.lower()
    if "campus" in s: return "City"
    if "factory" in s: return "Factory"
    if "farm" in s: return "Rural"
    if "gress" in s or "grass" in s: return "Natural"
    return SCENE_CAT_MAP.get(scene_name.strip(), "Uncategorized")

# ================= 3. 核心计算逻辑 =================

def align_scale_shift_torch(pred, target, mask):
    safe_mask = mask & torch.isfinite(pred) & torch.isfinite(target)
    if safe_mask.sum() < 10: return pred
    
    t_valid = target[safe_mask]
    p_valid = pred[safe_mask]
    
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

def compute_metrics_detailed(gt, pred, valid_mask):
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
    
    pixel_mask = valid_mask & torch.isfinite(gt) & torch.isfinite(pred) & (gt_c > 1e-7)
    mask_f = pixel_mask.float()
    valid_pixels = mask_f.sum()
    
    if valid_pixels < 10:
        return torch.zeros(12, device=gt.device)
        
    diff = gt_c - pred_c
    diff_sq = diff ** 2
    
    abs_rel = (torch.abs(diff) / gt_c * mask_f).sum()
    sq_rel = (diff_sq / gt_c * mask_f).sum()
    rmse_sq = (diff_sq * mask_f).sum()
    
    log_diff = torch.log(gt_c) - torch.log(pred_c)
    log_diff = torch.nan_to_num(log_diff, 0.0)
    rmse_log_sq = (log_diff ** 2 * mask_f).sum()
    
    thresh = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    thresh = torch.nan_to_num(thresh, posinf=999.0)
    
    a1 = ((thresh < 1.25) * mask_f).sum()
    a2 = ((thresh < 1.25 ** 2) * mask_f).sum()
    a3 = ((thresh < 1.25 ** 3) * mask_f).sum()
    
    # Numpy 指标
    p_np = pred_c[pixel_mask].cpu().numpy()
    g_np = gt_c[pixel_mask].cpu().numpy()
    
    ld = np.log(g_np) - np.log(p_np)
    si_log = np.sqrt(np.mean(ld**2) - np.mean(ld)**2 + 1e-10)
    
    p_norm = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8)
    g_norm = (g_np - g_np.min()) / (g_np.max() - g_np.min() + 1e-8)
    n_rmse = np.sqrt(np.mean((p_norm - g_norm)**2))
    
    if len(p_np) > 5000:
        idx = np.random.choice(len(p_np), 5000, replace=False)
        p_np, g_np = p_np[idx], g_np[idx]
        
    if np.std(p_np) < 1e-7 or np.std(g_np) < 1e-7:
        spear = 0.0
    else:
        try:
            s_val, _ = spearmanr(p_np, g_np)
            spear = 0.0 if np.isnan(s_val) else s_val
        except: spear = 0.0

    # 这里的 1.0 是 ImgCount
    return torch.tensor([
        abs_rel, sq_rel, rmse_sq, rmse_log_sq, a1, a2, a3, 
        valid_pixels, si_log, spear, n_rmse, 1.0
    ], device=gt.device)

def compute_final_mean(sums):
    """返回长度为 10 的数组"""
    pix = sums[7]
    img = sums[11]
    if pix <= 0: return np.zeros(10)
    
    # 0-6: Pixel based
    m_pix = np.array([
        sums[0]/pix, sums[1]/pix, 
        np.sqrt(sums[2]/pix), np.sqrt(sums[3]/pix),
        sums[4]/pix, sums[5]/pix, sums[6]/pix
    ])
    # 7-9: Image based (对应 sums[8], sums[9], sums[10])
    m_img = np.array([sums[8]/img, sums[9]/img, sums[10]/img]) if img > 0 else np.zeros(3)
    
    # 结果包含 [AbsRel, SqRel, RMSE, RMSlg, a1, a2, a3, SI-Log, Spear, N-RMSE]
    # 索引:     0       1      2     3      4   5   6   7       8      9
    return np.concatenate([m_pix, m_img])

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
                
                fov = float(row[5]) if pd.notna(row[5]) else 60.0
                pitch = float(row[7]) if pd.notna(row[7]) else 0.0
                focal = (4000 / 2) / np.tan(np.deg2rad(fov) / 2)
                
                p = self._find(Path(pred_root), scene, [stem1, stem2], False)
                g = self._find(Path(gt_root), scene, [stem1, stem2], True)
                if p and g:
                    self.data.append({
                        "pred": p, "gt": g, 
                        "meta": {"scene": scene, "pitch": pitch, "focal": focal}
                    })
        except: pass

    def _find(self, root, scene, stems, is_gt):
        cands = [f"{s}.npy" for s in stems]
        if is_gt: cands += ["depth.npy", "gt.npy"]
        else: cands += ["depth.npy"]
        dirs = [root/scene, root, root/scene/stems[0], root/stems[0]]
        for d in dirs:
            if d.exists():
                for c in cands:
                    if (d/c).exists(): return d/c
        return None

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        try:
            return {
                "pred": torch.from_numpy(np.load(self.data[idx]["pred"])),
                "gt": torch.from_numpy(np.load(self.data[idx]["gt"])),
                "meta": self.data[idx]["meta"]
            }
        except: return None

def collate_fn(b):
    b = [x for x in b if x is not None]
    return b if b else None

# ================= 5. 统计与报告 =================

class StatsTracker:
    def __init__(self):
        # 累加器大小是 12
        self.acc_overall = torch.zeros(12).to(device)
        self.acc_scene = {}
        self.acc_cat = {}
        self.acc_pitch = {}
        self.acc_focal = {}
        self.counts = {'scene': {}, 'cat': {}, 'pitch': {}, 'focal': {}}

    def update(self, vals, meta):
        self.acc_overall += vals
        
        def _up(d_acc, d_cnt, key):
            if key not in d_acc:
                d_acc[key] = torch.zeros_like(vals)
                d_cnt[key] = set()
            d_acc[key] += vals
            d_cnt[key].add(meta['scene'])

        _up(self.acc_scene, self.counts['scene'], meta['scene'])
        
        cat = get_scene_category(meta['scene'])
        _up(self.acc_cat, self.counts['cat'], cat)
        
        p = meta['pitch']
        if p < -75: pk = "-90to-75"
        elif p < -60: pk = "-75to-60"
        elif p < -45: pk = "-60to-45"
        else: pk = "-45to-0"
        _up(self.acc_pitch, self.counts['pitch'], pk)
        
        f = meta['focal']
        fk = f"{int(f//FOCAL_STEP)*FOCAL_STEP}-{(int(f//FOCAL_STEP)+1)*FOCAL_STEP}"
        _up(self.acc_focal, self.counts['focal'], fk)

def format_line(name, m, count_img, count_scn):
    m = np.nan_to_num(m)
    return "{:<20} | {:>4} | {:>4} | {:>6.4f} | {:>6.2f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:>6.3f} | {:>6.3f} | {:>6.3f} |".format(
        str(name)[-20:], int(count_img), int(count_scn),
        m[0], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9]
    )

def format_header():
    return "{:<20} | {:>4} | {:>4} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} || {:>6} | {:>6} | {:>6} |".format(
        "Category", "Img#", "Scn#", "AbsRel", "RMSE", "RMSlg", "a1", "a2", "a3", "SI-Log", "Spear", "N-RMSE")

# ================= 6. 主程序 =================

if __name__ == "__main__":
    print(f"🚀 Starting Detailed Batch Eval ({len(TASKS)} Tasks)")
    tracker = StatsTracker()
    total_imgs = 0
    
    for task in TASKS:
        print(f"Processing {task['name']}...")
        ds = RobustDataset(task['csv_path'], task['pred_root'], task['gt_root'])
        if len(ds) == 0: continue
        
        dl = DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS, collate_fn=collate_fn)
        
        for batch in tqdm(dl, leave=False):
            if not batch: continue
            item = batch[0]
            p = item['pred'].to(device).squeeze()
            g = item['gt'].to(device).squeeze()
            meta = item['meta']
            
            if p.shape != g.shape:
                p = F.interpolate(p[None,None], size=g.shape, mode='bilinear').squeeze()
            
            mask = (g > MIN_EVAL_DEPTH) & (g < MAX_EVAL_DEPTH) & (p > 0)
            if USE_ALIGNMENT:
                p = align_scale_shift_torch(p, g, mask)
            
            res_tensor = compute_metrics_detailed(g, p, mask)
            
            if res_tensor[11] > 0: # 这里的 index 11 是 ImgCount，在累加器中是存在的
                tracker.update(res_tensor, meta)
                total_imgs += 1

    if total_imgs == 0:
        print("❌ No valid data processed.")
        exit()

    lines = []
    header = format_header()
    sep = "-" * len(header)
    lines += ["DETAILED STRUCTURAL REPORT", f"Date: {datetime.now()}", f"Total Images: {total_imgs}", sep, header, sep]
    
    # 1. Overall
    ov_res = compute_final_mean(tracker.acc_overall.cpu().numpy())
    all_scns = len(tracker.counts['scene'])
    lines.append(format_line("OVERALL", ov_res, total_imgs, all_scns))
    lines.append(sep)
    
    # 2. Category
    lines.append(">>> BY CATEGORY")
    cat_order = ["City", "Factory", "Rural", "Natural", "Uncategorized"]
    for k in cat_order:
        if k in tracker.acc_cat:
            res = compute_final_mean(tracker.acc_cat[k].cpu().numpy())
            # === [修复点] ===
            # 不要去 res 里找图片数，而是直接去 tracker.acc_cat 里拿 (index 11)
            img_count = int(tracker.acc_cat[k][11].item())
            lines.append(format_line(k, res, img_count, len(tracker.counts['cat'][k])))
    lines.append(sep)
    
    # 3. Pitch
    lines.append(">>> BY PITCH")
    for k in sorted(tracker.acc_pitch.keys()):
        res = compute_final_mean(tracker.acc_pitch[k].cpu().numpy())
        img_count = int(tracker.acc_pitch[k][11].item())
        lines.append(format_line(k, res, img_count, len(tracker.counts['pitch'][k])))
    lines.append(sep)
    
    # 4. Focal
    lines.append(">>> BY FOCAL")
    keys = sorted(tracker.acc_focal.keys(), key=lambda x: int(x.split('-')[0]))
    for k in keys:
        res = compute_final_mean(tracker.acc_focal[k].cpu().numpy())
        img_count = int(tracker.acc_focal[k][11].item())
        lines.append(format_line(k, res, img_count, len(tracker.counts['focal'][k])))
    lines.append(sep)
    
    # 5. Scene
    lines.append(">>> BY SCENE (Top 20 Spearman)")
    scene_list = []
    for k in tracker.acc_scene:
        res = compute_final_mean(tracker.acc_scene[k].cpu().numpy())
        img_count = int(tracker.acc_scene[k][11].item())
        scene_list.append((k, res, img_count))
    
    scene_list.sort(key=lambda x: x[1][8], reverse=True) 
    for k, res, img_count in scene_list[:20]:
         lines.append(format_line(k, res, img_count, 1))

    print("\n".join(lines))
    
    out_path = os.path.join(BASE_PRED_DIR, "Final_Detailed_Structural_Report.txt")
    with open(out_path, "w") as f: f.write("\n".join(lines))
    print(f"\n✅ Report Saved: {out_path}")