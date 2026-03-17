# import os
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
# import argparse
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# parser = argparse.ArgumentParser(description="Step 3: Evaluation with Viz")
# parser.add_argument("--pred_root", type=str, required=True, help="预测结果目录")
# parser.add_argument("--gt_root", type=str, required=True, help="GT 真值目录")
# parser.add_argument("--csv_path", type=str, required=True, help="CSV 索引文件")
# # === 新增参数 ===
# parser.add_argument("--viz_dir", type=str, default=None, help="可视化保存路径，不填则默认保存在 pred_root/viz_results")
# parser.add_argument("--viz_interval", type=int, default=50, help="每隔多少张图保存一张可视化结果")

# args = parser.parse_args()

# PRED_ROOT = args.pred_root
# VAL_GT_ROOTS = [args.gt_root] 
# MASTER_CSV = args.csv_path

# # 可视化输出路径
# VIZ_DIR = args.viz_dir if args.viz_dir else os.path.join(PRED_ROOT, "viz_results")
# os.makedirs(VIZ_DIR, exist_ok=True)

# # 2. 评估参数
# MIN_EVAL_DEPTH = 1e-3
# MAX_EVAL_DEPTH = 400
# USE_MEDIAN_SCALING = False
# BATCH_SIZE = 1 
# NUM_WORKERS = 4 # 绘图时 worker太多可能导致matplotlib后端冲突，建议调小
# FOCAL_STEP = 400

# DEPTH_BINS = [(0, 50), (50, 120), (120, 250), (250, 400), (400, 500)]
# BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]
# ALL_LABELS = BIN_LABELS + ["Overall"]

# # ================= 场景映射 (保持不变) =================
# _RURAL_LIST = ["ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output", "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled", "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled", "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled", "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output", "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled"]
# _NATURAL_LIST = ["lewis-output", "park5", "park13", "park14", "park10", "park0", "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled", "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled", "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled", "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled", "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled", "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"]
# _CITY_LIST = ["yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"]
# _FACTORY_LIST = ["BC2", "BC1", "S-output"]

# SCENE_CAT_MAP = {}
# for s in _RURAL_LIST: SCENE_CAT_MAP[s] = "Rural"
# for s in _NATURAL_LIST: SCENE_CAT_MAP[s] = "Natural"
# for s in _CITY_LIST: SCENE_CAT_MAP[s] = "City"
# for s in _FACTORY_LIST: SCENE_CAT_MAP[s] = "Factory"

# def get_scene_category(scene_name):
#     return SCENE_CAT_MAP.get(scene_name.strip(), "Uncategorized")

# warnings.filterwarnings("ignore")

# # ================= 核心逻辑 =================

# def align_scale_shift_torch(pred, target, mask):
#     # (保持原本的对齐逻辑不变)
#     safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
#     t_valid = target[safe_mask]
#     p_valid = pred[safe_mask]
#     if len(t_valid) < 10:
#         if len(t_valid) > 0:
#             ratio = torch.median(t_valid) / (torch.median(p_valid) + 1e-8)
#             return pred * ratio
#         else: return pred 
#     ones = torch.ones_like(p_valid)
#     A = torch.stack([p_valid, ones], dim=1)
#     s, t = 1.0, 0.0
#     success = False
#     try:
#         solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
#         s = solution[0].item()
#         t = solution[1].item()
#         if np.isnan(s) or np.isnan(t) or np.isinf(s) or np.isinf(t): raise ValueError()
#         if s <= 1e-4: raise ValueError()
#         success = True
#     except Exception: success = False
#     if not success:
#         med_t = torch.median(t_valid)
#         med_p = torch.median(p_valid)
#         if med_p > 1e-6:
#             s = (med_t / med_p).item()
#             t = 0.0
#         else: s, t = 1.0, 0.0
#     return pred * s + t

# def compute_errors_torch_bins(gt, pred, valid_mask):
#     # (保持原本的计算逻辑不变)
#     if pred.dim() == 3 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
#     if gt.dim() == 3 and gt.shape[-1] == 1: gt = gt.squeeze(-1)
    
#     gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
#     pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
#     rmse_map = (gt - pred) ** 2
#     rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
#     abs_rel_map = torch.abs(gt - pred) / gt_c
#     sq_rel_map = ((gt - pred) ** 2) / gt_c
#     thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
#     all_ranges = DEPTH_BINS + [("Overall", "Overall")]
#     batch_results = []
    
#     for (b_min, b_max) in all_ranges:
#         if b_min == "Overall":
#             current_mask = valid_mask
#         else:
#             current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
            
#         mask_f = current_mask.float()
#         valid_pixel_count = mask_f.sum(dim=[1, 2])
        
#         a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
#         a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
#         a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
#         rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
#         rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
#         abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
#         sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])

#         si_log_list, spearman_list, norm_rmse_list = [], [], []
        
#         for b in range(gt.shape[0]):
#             m = current_mask[b]
#             if m.sum() < 10:
#                 si_log_list.append(0.0); spearman_list.append(0.0); norm_rmse_list.append(0.0)
#                 continue
#             g_valid = gt_c[b][m]
#             p_valid = pred_c[b][m]
#             log_diff = torch.log(g_valid) - torch.log(p_valid)
#             si_log_val = torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2 + 1e-8)
#             si_log_list.append(si_log_val.item())
            
#             p_np = p_valid.detach().cpu().numpy()
#             g_np = g_valid.detach().cpu().numpy()
#             if len(p_np) > 5000:
#                 idx = np.random.choice(len(p_np), 5000, replace=False)
#                 p_np = p_np[idx]; g_np = g_np[idx]
            
#             if np.std(p_np) < 1e-6 or np.std(g_np) < 1e-6: spearman_val = 0.0
#             else:
#                 spearman_val, _ = spearmanr(p_np, g_np)
#                 if np.isnan(spearman_val): spearman_val = 0.0
#             spearman_list.append(spearman_val)
            
#             p_norm = (p_valid - p_valid.min()) / (p_valid.max() - p_valid.min() + 1e-8)
#             g_norm = (g_valid - g_valid.min()) / (g_valid.max() - g_valid.min() + 1e-8)
#             norm_rmse_val = torch.sqrt(torch.mean((p_norm - g_norm)**2))
#             norm_rmse_list.append(norm_rmse_val.item())

#         si_log_t = torch.tensor(si_log_list, device=gt.device)
#         spearman_t = torch.tensor(spearman_list, device=gt.device)
#         norm_rmse_t = torch.tensor(norm_rmse_list, device=gt.device)
#         valid_img_count = (valid_pixel_count > 10).float()

#         bin_res = torch.stack([abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count, si_log_t, spearman_t, norm_rmse_t, valid_img_count], dim=1)
#         batch_results.append(bin_res)
        
#     return torch.stack(batch_results, dim=1)

# def compute_metrics_from_sums(sums):
#     total_valid_pixels = sums[7]
#     total_valid_imgs = sums[11]
#     if total_valid_pixels <= 0: return np.zeros(11)
    
#     pixel_metrics = np.array([
#         sums[0]/total_valid_pixels, sums[1]/total_valid_pixels,
#         np.sqrt(sums[2]/total_valid_pixels), np.sqrt(sums[3]/total_valid_pixels),
#         sums[4]/total_valid_pixels, sums[5]/total_valid_pixels, sums[6]/total_valid_pixels
#     ])
#     if total_valid_imgs > 0:
#         img_metrics = np.array([sums[8]/total_valid_imgs, sums[9]/total_valid_imgs, sums[10]/total_valid_imgs])
#     else: img_metrics = np.zeros(3)
#     return np.concatenate([pixel_metrics, img_metrics])

# # ================= 🎨 可视化模块 (新增) =================

# def visualize_sample(rgb, gt, pred, mask, save_path, meta):
#     """
#     生成 2x3 的可视化图表：
#     1. RGB | 2. GT Depth | 3. Pred Depth
#     4. AbsRel Heatmap | 5. Alpha Classification | 6. Log Difference
#     """
#     # 转换为 numpy
#     if torch.is_tensor(rgb): rgb = rgb.cpu().numpy().transpose(1, 2, 0)
#     if torch.is_tensor(gt): gt = gt.cpu().numpy()
#     if torch.is_tensor(pred): pred = pred.cpu().numpy()
#     if torch.is_tensor(mask): mask = mask.cpu().numpy()

#     # 归一化 RGB (如果是 float [0,1])
#     if rgb.max() > 1.0: rgb = rgb / 255.0
    
#     # 设置显示范围
#     valid_gt = gt[mask]
#     if len(valid_gt) == 0: return # 空数据不画
#     vmin, vmax = valid_gt.min(), valid_gt.max()

#     # 计算指标用于绘图
#     gt_safe = np.maximum(gt, 1e-3)
#     pred_safe = np.maximum(pred, 1e-3)
    
#     # 1. AbsRel Map
#     abs_rel = np.abs(gt - pred) / gt_safe
#     abs_rel[~mask] = 0
    
#     # 2. Alpha Map (Your Request)
#     thresh = np.maximum(gt_safe / pred_safe, pred_safe / gt_safe)
    
#     # 创建离散颜色图
#     # 0: Invalid (Black), 1: a1 (Green), 2: a2 (Yellow), 3: a3 (Orange), 4: Bad (Red)
#     alpha_map = np.zeros_like(gt, dtype=np.uint8) 
    
#     # 先填满 Bad
#     alpha_map[mask] = 4 
#     # 逐步覆盖更好的
#     alpha_map[mask & (thresh < 1.25**3)] = 3
#     alpha_map[mask & (thresh < 1.25**2)] = 2
#     alpha_map[mask & (thresh < 1.25)] = 1
#     alpha_map[~mask] = 0 # Invalid
    
#     # 自定义 Colormap for Alpha
#     # colors: Black, Green, Yellow, Orange, Red
#     alpha_colors = ['black', '#00cc00', '#ffdb14', '#ff8c00', '#ff0000']
#     cmap_alpha = mcolors.ListedColormap(alpha_colors)
#     bounds = [0, 0.9, 1.9, 2.9, 3.9, 5]
#     norm_alpha = mcolors.BoundaryNorm(bounds, cmap_alpha.N)

#     # 3. Log Difference Map (Reflects structural error)
#     # log(pred) - log(gt): >0 (Red) means pred is further, <0 (Blue) means pred is closer
#     log_diff = np.log(pred_safe) - np.log(gt_safe)
#     log_diff[~mask] = np.nan

#     # --- Plotting ---
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#     plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
#     # Subplot 1: RGB
#     axes[0, 0].imshow(rgb)
#     axes[0, 0].set_title(f"RGB: {meta['scene']}\nPitch:{meta['pitch']:.1f}", fontsize=10)
#     axes[0, 0].axis('off')

#     # Subplot 2: GT
#     im2 = axes[0, 1].imshow(gt, cmap='magma_r', vmin=vmin, vmax=vmax)
#     axes[0, 1].set_title("Ground Truth (m)", fontsize=10)
#     axes[0, 1].axis('off')
#     plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

#     # Subplot 3: Pred
#     im3 = axes[0, 2].imshow(pred, cmap='magma_r', vmin=vmin, vmax=vmax)
#     axes[0, 2].set_title("Predicted Depth (m)", fontsize=10)
#     axes[0, 2].axis('off')
#     plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

#     # Subplot 4: AbsRel Error
#     im4 = axes[1, 0].imshow(abs_rel, cmap='turbo', vmin=0, vmax=0.5) # vmax设为0.5让小误差更明显
#     axes[1, 0].set_title("AbsRel Error Heatmap\n(Red=High Error)", fontsize=10)
#     axes[1, 0].axis('off')
#     plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

#     # Subplot 5: Alpha Classification (User Request)
#     im5 = axes[1, 1].imshow(alpha_map, cmap=cmap_alpha, norm=norm_alpha, interpolation='nearest')
#     axes[1, 1].set_title("Alpha Metrics Visualization", fontsize=10)
#     axes[1, 1].axis('off')
#     # Custom Legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='#00cc00', label=r'$\alpha_1 (<1.25)$'),
#         Patch(facecolor='#ffdb14', label=r'$\alpha_2 (<1.25^2)$'),
#         Patch(facecolor='#ff8c00', label=r'$\alpha_3 (<1.25^3)$'),
#         # 🔥 修改了这里：将 \ge 改为了 \geq
#         Patch(facecolor='#ff0000', label=r'Bad ($\geq 1.25^3$)'),
#         Patch(facecolor='black', label='Invalid')
#     ]
#     axes[1, 1].legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.6)

#     # Subplot 6: Log Difference
#     # Diverging colormap: Blue(Close) - White(Accurate) - Red(Far)
#     limit = max(abs(np.nanmin(log_diff)), abs(np.nanmax(log_diff)))
#     limit = min(limit, 1.0) # Clamp visualization range
#     im6 = axes[1, 2].imshow(log_diff, cmap='seismic', vmin=-limit, vmax=limit)
#     axes[1, 2].set_title("Log Difference (Structure)\nBlue:Pred<GT | Red:Pred>GT", fontsize=10)
#     axes[1, 2].axis('off')
#     plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

#     plt.suptitle(f"Scene: {meta['scene']} | Img: {os.path.basename(save_path)}", fontsize=12)
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()

# # ================= 改进 Dataset (支持 RGB 读取) =================

# class CSVIndexDataset(Dataset):
#     def __init__(self, csv_path, pred_root, gt_roots):
#         self.df = pd.read_csv(csv_path)
#         self.val_df = self.df[self.df['Split'] == 'val'].reset_index(drop=True)
#         self.pred_root = Path(pred_root)
#         self.gt_roots = [Path(p) for p in gt_roots]
#         print(f"Dataset initialized. Val samples: {len(self.val_df)}")

#     def __len__(self): return len(self.val_df)
    
#     def find_file(self, roots, scene, sub, stems, exts=['.npy']):
#         for root in roots:
#             base = root / scene / sub
#             if not base.exists(): continue
#             for stem in stems:
#                 for ext in exts:
#                     cand = base / f"{stem}{ext}"
#                     if cand.exists(): return cand
#                     # Try case-insensitive
#                     try:
#                         lower_map = {f.lower(): f for f in os.listdir(base)}
#                         if f"{stem}{ext}".lower() in lower_map:
#                             return base / lower_map[f"{stem}{ext}".lower()]
#                     except: pass
#         return None

#     def __getitem__(self, idx):
#         row = self.val_df.iloc[idx]
#         scene = str(row['Scene_Name'])
#         renamed = str(row['Renamed_Image'])
#         original = str(row['Original_Filename'])
#         stems = []
#         if pd.notna(renamed): stems.append(os.path.splitext(renamed)[0])
#         if pd.notna(original): stems.append(os.path.splitext(original)[0])
        
#         # 1. Load GT and Pred
#         gt_path = self.find_file(self.gt_roots, scene, "depth", stems, ['.npy'])
#         pred_path = self.find_file([self.pred_root], scene, "", stems, ['.npy'])
        
#         # 2. Try Load RGB (Assuming standard folder structure or inside GT root)
#         # Search in 'images', 'rgb', or root
#         rgb_subfolders = ["images", "rgb", "img", ""]
#         rgb_path = None
#         for sub in rgb_subfolders:
#             rgb_path = self.find_file(self.gt_roots, scene, sub, stems, ['.jpg', '.png', '.jpeg'])
#             if rgb_path: break
            
#         if gt_path is None or pred_path is None: return None 
            
#         try:
#             gt = np.load(gt_path).astype(np.float32)
#             pred = np.load(pred_path).astype(np.float32)
            
#             # Load RGB
#             if rgb_path:
#                 rgb = cv2.imread(str(rgb_path))
#                 if rgb is not None:
#                     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
#                 else:
#                     rgb = np.zeros((*gt.shape, 3), dtype=np.uint8) # Fallback black
#             else:
#                 rgb = np.zeros((*gt.shape, 3), dtype=np.uint8) # Fallback black

#             # Resize RGB if mismatch
#             if rgb.shape[:2] != gt.shape:
#                 rgb = cv2.resize(rgb, (gt.shape[1], gt.shape[0]))

#             focal_val = row.get('FocalLength_New', row.get('FocalLength(px)', 0))
#             meta = {
#                 'scene': scene,
#                 'pitch': row.get('Visual_Pitch(deg)', 0),
#                 'focal': focal_val,
#                 'idx': idx # for viz naming
#             }
#             return torch.from_numpy(pred), torch.from_numpy(gt), torch.from_numpy(rgb).permute(2,0,1), meta
#         except Exception as e:
#             # print(f"Error loading {scene}: {e}")
#             return None

# def collate_fn(batch):
#     batch = [b for b in batch if b is not None]
#     if len(batch) == 0: return None
#     preds = [b[0] for b in batch]
#     gts = [b[1] for b in batch]
#     rgbs = [b[2] for b in batch]
#     metas = [b[3] for b in batch]
#     return preds, gts, rgbs, metas

# # ================= 辅助函数 =================
# def format_line(name, m, count_img=0, count_scn=0, indent=0):
#     sp = " " * indent
#     m = np.nan_to_num(m)
#     return "{:<20} | {:>4} | {:>4} | {:>6.4f} | {:>6.2f} | {:>6.4f} | {:>6.4f} | {:>6.4f} | {:>6.4f} || {:>6.4f} | {:>6.4f} | {:>6.4f} |".format(
#         sp + str(name)[-20+indent:], count_img, count_scn,
#         m[0], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9]
#     )

# def format_header():
#     return "{:<20} | {:>4} | {:>4} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} || {:>6} | {:>6} | {:>6} |".format(
#         "Category", "Img#", "Scn#", "AbsRel", "RMSE", "RMSElg", "a1", "a2", "a3", "SI-Log", "Spear", "N-RMSE")

# # ================= 主流程 =================

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     dataset = CSVIndexDataset(MASTER_CSV, PRED_ROOT, VAL_GT_ROOTS)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
#     acc_overall = torch.zeros(len(ALL_LABELS), 12).to(device)
#     acc_pitch = {}
#     acc_focal = {}
#     acc_scene = {}
#     acc_category = {}
    
#     count_pitch = {}; count_focal = {}; count_scene = {}; count_category = {}
#     total_imgs = 0
    
#     print(f"🚀 开始全量评估 + 可视化 (Interval={args.viz_interval})...")
#     print(f"📂 可视化结果将保存至: {VIZ_DIR}")
    
#     for batch in tqdm(loader):
#         if batch is None: continue
#         preds, gts, rgbs, metas = batch
        
#         for i in range(len(preds)):
#             pred = preds[i].to(device)
#             gt = gts[i].to(device)
#             rgb = rgbs[i] # CPU tensor
#             meta = metas[i]
            
#             if pred.dim() == 3 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
#             if gt.dim() == 3 and gt.shape[-1] == 1: gt = gt.squeeze(-1)
#             if pred.shape != gt.shape:
#                 pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()
            
#             mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH)
#             if mask.sum() < 10: continue
            
#             if USE_MEDIAN_SCALING:
#                 pred = align_scale_shift_torch(pred, gt, mask)
            
#             pred.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
            
#             # --- 核心指标计算 ---
#             batch_sums = compute_errors_torch_bins(gt.unsqueeze(0), pred.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            
#             total_imgs += 1
            
#             # --- 🎨 可视化抽样逻辑 ---
#             if total_imgs % args.viz_interval == 0:
#                 viz_name = f"{meta['scene']}_{meta['idx']}.png"
#                 viz_path = os.path.join(VIZ_DIR, viz_name)
#                 # 使用 CPU 数据进行绘图，避免阻塞 GPU
#                 visualize_sample(rgb, gt.cpu(), pred.cpu(), mask.cpu(), viz_path, meta)
            
#             # --- 统计累加 ---
#             acc_overall += batch_sums
            
#             def update_stats(acc_dict, count_dict, key, vals, scene_name):
#                 if key not in acc_dict:
#                     acc_dict[key] = torch.zeros_like(vals)
#                     count_dict[key] = {'img': 0, 'scn': set()}
#                 acc_dict[key] += vals
#                 count_dict[key]['img'] += 1
#                 count_dict[key]['scn'].add(scene_name)

#             p_val = float(meta['pitch'])
#             p_key = "-90to-75" if p_val < -75 else "-75to-60" if p_val < -60 else "-60to-45" if p_val < -45 else "-45to-30" if p_val < -30 else "-30to-0"
#             update_stats(acc_pitch, count_pitch, p_key, batch_sums, meta['scene'])
            
#             f_val = meta['focal']
#             f_start = int(f_val // FOCAL_STEP) * FOCAL_STEP
#             f_key = f"{f_start}-{f_start + FOCAL_STEP}"
#             update_stats(acc_focal, count_focal, f_key, batch_sums, meta['scene'])
            
#             update_stats(acc_scene, count_scene, meta['scene'], batch_sums, meta['scene'])
            
#             cat_name = get_scene_category(meta['scene'])
#             update_stats(acc_category, count_category, cat_name, batch_sums, meta['scene'])

#     # ================= 生成报告 (保持不变) =================
#     lines = []
#     header = format_header()
#     sep = "-" * len(header)
#     lines += ["EVALUATION REPORT", f"Date: {datetime.now()}", "="*130, header, sep]
    
#     ov_res = compute_metrics_from_sums(acc_overall[-1].cpu().numpy())
#     all_scenes = set()
#     for v in count_scene.values(): all_scenes.update(v['scn'])
#     lines.append(format_line("OVERALL", ov_res, total_imgs, len(all_scenes)))
#     lines.append(sep)
    
#     lines.append(">>> BY SCENE CATEGORY")
#     cat_order = ["Rural", "Natural", "City", "Factory", "Uncategorized"]
#     existing_cats = [c for c in cat_order if c in acc_category]
#     for k in existing_cats:
#         lines.append(format_line(k, compute_metrics_from_sums(acc_category[k][-1].cpu().numpy()), count_category[k]['img'], len(count_category[k]['scn'])))
#     lines.append(sep)
    
#     lines.append(">>> BY PITCH")
#     sorted_keys = sorted(acc_pitch.keys())
#     for k in sorted_keys:
#         lines.append(format_line(k, compute_metrics_from_sums(acc_pitch[k][-1].cpu().numpy()), count_pitch[k]['img'], len(count_pitch[k]['scn'])))
#     lines.append(sep)
    
#     lines.append(f">>> BY FOCAL")
#     sorted_focal = sorted(acc_focal.keys(), key=lambda x: int(x.split('-')[0]))
#     for k in sorted_focal:
#         lines.append(format_line(k, compute_metrics_from_sums(acc_focal[k][-1].cpu().numpy()), count_focal[k]['img'], len(count_focal[k]['scn'])))
#     lines.append(sep)
    
#     lines.append(">>> ALL SCENES (Sorted by Spearman)")
#     scene_list = []
#     for k in acc_scene.keys():
#         scene_list.append((k, compute_metrics_from_sums(acc_scene[k][-1].cpu().numpy()), count_scene[k]))
#     scene_list.sort(key=lambda x: x[1][8], reverse=True)
#     for k, res, cnt in scene_list:
#         lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
    
#     report_text = "\n".join(lines)
#     print(report_text)
    
#     out_file = os.path.join(PRED_ROOT, "Final_Report_Viz.txt")
#     with open(out_file, "w") as f: f.write(report_text)
#     print(f"\n✅ 报告已保存至: {out_file}")
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
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ================= ⚙️ 配置参数 =================

parser = argparse.ArgumentParser(description="Step 3: Evaluation with Viz")
parser.add_argument("--pred_root", type=str, required=True, help="预测结果目录")
parser.add_argument("--gt_root", type=str, required=True, help="GT 真值目录")
parser.add_argument("--csv_path", type=str, required=True, help="CSV 索引文件")
parser.add_argument("--viz_dir", type=str, default=None, help="可视化保存路径")
parser.add_argument("--viz_interval", type=int, default=50, help="采样间隔")

args = parser.parse_args()

PRED_ROOT = args.pred_root
VAL_GT_ROOTS = [args.gt_root] 
MASTER_CSV = args.csv_path

# 可视化输出路径
VIZ_DIR = args.viz_dir if args.viz_dir else os.path.join(PRED_ROOT, "viz_results")
os.makedirs(VIZ_DIR, exist_ok=True)

# 缺失文件日志 (帮助你 Debug 为什么只加载了一部分)
MISSING_LOG_PATH = os.path.join(PRED_ROOT, "missing_files_debug.txt")
# 清空旧日志
with open(MISSING_LOG_PATH, "w") as f:
    f.write(f"Missing Files Log - {datetime.now()}\n{'='*50}\n")

# 评估参数
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 400
USE_MEDIAN_SCALING = False
BATCH_SIZE = 1 
NUM_WORKERS = 4 
FOCAL_STEP = 400

DEPTH_BINS = [(0, 50), (50, 120), (120, 250), (250, 400), (400, 500)]
BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]
ALL_LABELS = BIN_LABELS + ["Overall"]

# ================= 场景映射 =================
_RURAL_LIST = ["ainterval5_AMtown01_cropped_downsampled", "interval5_AMtown03_cropped_downsampled", "caliterra-output", "interval5_HKairport01_cropped_downsampled", "interval5_HKairport_GNSS_Evening_cropped_downsampled", "interval5_HKairport_GNSS02_cropped_downsampled", "interval5_HKairport_GNSS03_cropped_downsampled", "interval5_HKairport03_cropped_downsampled", "interval5_HKairport_GNSS01_cropped_downsampled", "R-PHD-output", "ODM1-output", "ODM2-output", "ODM32-output", "ODM34-output", "ainterval5_AMtown02_cropped_downsampled", "interval5_HKairport02_cropped_downsampled"]
_NATURAL_LIST = ["lewis-output", "park5", "park13", "park14", "park10", "park0", "interval5_AMvalley02_cropped_downsampled", "interval5_AMvalley01_cropped_downsampled", "interval5_HKisland_GNSS_Evening_cropped_downsampled", "interval5_HKisland_GNSS03_cropped_downsampled", "interval5_HKisland_GNSS02_cropped_downsampled", "interval5_HKisland_GNSS01_cropped_downsampled", "interval5_HKisland03_cropped_downsampled", "interval5_HKisland01_cropped_downsampled", "bellus-output", "sceneca-output", "ainterval5_HKisland02_cropped_downsampled", "park8", "park9", "interval5_AMvalley03_cropped_downsampled", "ODM3-output", "ODM6-output"]
_CITY_LIST = ["yingrenshi", "hav", "upper", "sztu", "sziit", "polytech", "SMBU", "lfls", "lfls2", "longhua", "Artsci"]
_FACTORY_LIST = ["BC2", "BC1", "S-output"]

SCENE_CAT_MAP = {}
for s in _RURAL_LIST: SCENE_CAT_MAP[s] = "Rural"
for s in _NATURAL_LIST: SCENE_CAT_MAP[s] = "Natural"
for s in _CITY_LIST: SCENE_CAT_MAP[s] = "City"
for s in _FACTORY_LIST: SCENE_CAT_MAP[s] = "Factory"

def get_scene_category(scene_name):
    return SCENE_CAT_MAP.get(scene_name.strip(), "Uncategorized")

warnings.filterwarnings("ignore")

# ================= 核心逻辑 =================

def align_scale_shift_torch(pred, target, mask):
    safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    t_valid = target[safe_mask]
    p_valid = pred[safe_mask]
    if len(t_valid) < 10:
        return pred if len(t_valid) == 0 else pred * (torch.median(t_valid) / (torch.median(p_valid) + 1e-8))
    
    ones = torch.ones_like(p_valid)
    A = torch.stack([p_valid, ones], dim=1)
    try:
        solution = torch.linalg.lstsq(A, t_valid, driver='gels').solution
        s, t = solution[0].item(), solution[1].item()
        if s <= 1e-4 or np.isnan(s): raise ValueError()
    except:
        med_t, med_p = torch.median(t_valid), torch.median(p_valid)
        s = (med_t / med_p).item() if med_p > 1e-6 else 1.0
        t = 0.0
    return pred * s + t

def compute_errors_torch_bins(gt, pred, valid_mask):
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if gt.dim() == 3: gt = gt.squeeze(-1)
    
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
    rmse_map = (gt - pred) ** 2
    rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
    abs_rel_map = torch.abs(gt - pred) / gt_c
    sq_rel_map = ((gt - pred) ** 2) / gt_c
    thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
    all_ranges = DEPTH_BINS + [("Overall", "Overall")]
    batch_results = []
    
    for (b_min, b_max) in all_ranges:
        current_mask = valid_mask if b_min == "Overall" else valid_mask & (gt >= b_min) & (gt < b_max)
        mask_f = current_mask.float()
        valid_pixel_count = mask_f.sum(dim=[1, 2])
        
        a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
        a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
        a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
        rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
        rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
        abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
        sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])

        si_log_list, spearman_list, norm_rmse_list = [], [], []
        
        for b in range(gt.shape[0]):
            m = current_mask[b]
            if m.sum() < 10:
                si_log_list.append(0.0); spearman_list.append(0.0); norm_rmse_list.append(0.0)
                continue
            g_valid = gt_c[b][m]
            p_valid = pred_c[b][m]
            log_diff = torch.log(g_valid) - torch.log(p_valid)
            si_log_val = torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2 + 1e-8)
            si_log_list.append(si_log_val.item())
            
            p_np, g_np = p_valid.detach().cpu().numpy(), g_valid.detach().cpu().numpy()
            if len(p_np) > 5000:
                idx = np.random.choice(len(p_np), 5000, replace=False)
                p_np, g_np = p_np[idx], g_np[idx]
            
            if np.std(p_np) < 1e-6 or np.std(g_np) < 1e-6: spearman_val = 0.0
            else:
                spearman_val, _ = spearmanr(p_np, g_np)
                if np.isnan(spearman_val): spearman_val = 0.0
            spearman_list.append(spearman_val)
            
            p_norm = (p_valid - p_valid.min()) / (p_valid.max() - p_valid.min() + 1e-8)
            g_norm = (g_valid - g_valid.min()) / (g_valid.max() - g_valid.min() + 1e-8)
            norm_rmse_list.append(torch.sqrt(torch.mean((p_norm - g_norm)**2)).item())

        si_log_t = torch.tensor(si_log_list, device=gt.device)
        spearman_t = torch.tensor(spearman_list, device=gt.device)
        norm_rmse_t = torch.tensor(norm_rmse_list, device=gt.device)
        
        bin_res = torch.stack([abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count, si_log_t, spearman_t, norm_rmse_t, (valid_pixel_count > 10).float()], dim=1)
        batch_results.append(bin_res)
        
    return torch.stack(batch_results, dim=1)
def safe_load_rgb(path, target_hw=None):
    """
    智能读取图片，确保返回 (H, W, 3) 的 RGB 格式，处理灰度图、RGBA等情况
    """
    if path is None:
        # 如果没有路径，返回全黑图
        if target_hw:
            return np.zeros((target_hw[0], target_hw[1], 3), dtype=np.uint8)
        else:
            return np.zeros((512, 512, 3), dtype=np.uint8) # 默认大小

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) # 读取原格式
    
    if img is None:
        # 读失败了，返回全黑
        if target_hw:
            return np.zeros((target_hw[0], target_hw[1], 3), dtype=np.uint8)
        return np.zeros((512, 512, 3), dtype=np.uint8)

    # 1. 处理灰度图 (H, W) -> (H, W, 3)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # 2. 处理 RGBA (H, W, 4) -> (H, W, 3)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    # 3. 处理 BGR (H, W, 3) -> RGB
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. 强制 Resize (如果 GT 存在)
    if target_hw is not None:
        h, w = img.shape[:2]
        if h != target_hw[0] or w != target_hw[1]:
            img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
            
    return img
def compute_metrics_from_sums(sums):
    total_px = sums[7]
    total_imgs = sums[11]
    if total_px <= 0: return np.zeros(11)
    
    px_metrics = np.array([sums[0], sums[1], np.sqrt(sums[2]), np.sqrt(sums[3]), sums[4], sums[5], sums[6]]) / total_px
    px_metrics[2:4] *= np.sqrt(total_px) # Correct back sqrt
    px_metrics[2] = np.sqrt(sums[2]/total_px)
    px_metrics[3] = np.sqrt(sums[3]/total_px)
    
    img_metrics = np.array([sums[8], sums[9], sums[10]]) / total_imgs if total_imgs > 0 else np.zeros(3)
    return np.concatenate([px_metrics, img_metrics])

# ================= 🎨 修复后的可视化 =================



def visualize_sample(rgb, gt, pred, mask, save_path, meta):
    # ================= 📏 高精度航拍配置 =================
    # 1. 精度阈值设置
    TH_ELITE  = 1.05  # 5% 误差 (最严)
    TH_HIGH   = 1.15  # 15% 误差 (优秀)
    TH_STD    = 1.25  # 25% 误差 (及格 - 原Alpha1)
    
    # 2. 绝对误差豁免 (米)
    ABS_TOLERANCE = 1.0 
    
    # 3. 探测点
    PROBE_POINTS = [(0.5, 0.5), (0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8)]
    # =======================================================

    # --- 1. RGB 图像标准化 (修复全黑问题的关键) ---
    if torch.is_tensor(rgb): 
        # Tensor (C, H, W) -> Numpy (H, W, C)
        rgb = rgb.cpu().numpy().transpose(1, 2, 0)
    
    # 处理灰度图 (H, W) -> (H, W, 3)
    if rgb.ndim == 2:
        rgb = np.stack([rgb]*3, axis=-1)

    # 归一化到 [0.0, 1.0] 的 float32，避免 Matplotlib 显示异常
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    elif rgb.max() > 1.0:
        rgb = rgb.astype(np.float32) / 255.0
    
    # --- 2. 其他数据转 Numpy ---
    if torch.is_tensor(gt): gt = gt.cpu().numpy()
    if torch.is_tensor(pred): pred = pred.cpu().numpy()
    if torch.is_tensor(mask): mask = mask.cpu().numpy()

    # --- 3. 数据校验 ---
    valid_gt = gt[mask]
    if len(valid_gt) == 0: return

    # --- 4. 中位数对齐 (Strict 模式核心) ---
    valid_pred = pred[mask]
    if len(valid_pred) > 0:
        scale_shift = np.median(valid_gt) / (np.median(valid_pred) + 1e-8)
        pred_aligned = pred * scale_shift
    else:
        pred_aligned = pred

    # --- 5. 计算显示范围 (Clip) ---
    vmin = np.percentile(valid_gt, 2)
    vmax = np.percentile(valid_gt, 98)

    # --- 6. 计算指标 Map ---
    gt_safe = np.maximum(gt, 1e-3)
    pred_safe = np.maximum(pred_aligned, 1e-3)
    
    thresh = np.maximum(gt_safe / pred_safe, pred_safe / gt_safe) # Ratio
    abs_diff = np.abs(gt_safe - pred_safe) # Diff

    # 构建分类 Map (优先级：Invalid > Fail > Std > High > Elite > Noise)
    # 我们先全部设为 Fail (4 - Red)
    precision_map = np.full_like(gt, 4, dtype=np.uint8) 
    
    # 逐步覆盖更优的等级
    precision_map[mask & (thresh < TH_STD)] = 3   # Yellow
    precision_map[mask & (thresh < TH_HIGH)] = 2  # Cyan
    precision_map[mask & (thresh < TH_ELITE)] = 1 # Green
    
    # 豁免逻辑: 误差 < 1m 且不是 High/Elite 的，给豁免权 (Gray)
    # 意思是：虽然比例不对，但绝对值很小，不算 Fail/Std，算 Noise
    noise_mask = mask & (abs_diff < ABS_TOLERANCE) & (thresh >= TH_HIGH)
    precision_map[noise_mask] = 0 
    
    precision_map[~mask] = 5 # Invalid (Black)

    # 颜色配置
    colors = ['#999999', '#00cc00', '#00ffff', '#ffdb14', '#ff0000', 'black']
    cmap_prec = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_prec = mcolors.BoundaryNorm(bounds, cmap_prec.N)

    # Log Difference
    log_diff = np.log(pred_safe) - np.log(gt_safe)
    log_diff[~mask] = np.nan

    # --- 7. 绘图辅助函数 ---
    def draw_probes(ax, img, points, img_type='depth'):
        h, w = img.shape[:2]
        for i, (rx, ry) in enumerate(points):
            cx, cy = int(rx * w), int(ry * h)
            cx, cy = np.clip(cx, 0, w-1), np.clip(cy, 0, h-1)
            ax.scatter([cx], [cy], s=60, c='red', marker='+', zorder=10)
            
            if img_type == 'depth':
                val = img[cy, cx]
                txt = "Inv" if val < 1e-3 or not np.isfinite(val) else f"{val:.1f}m"
            elif img_type == 'ratio':
                val = thresh[cy, cx]
                txt = f"x{val:.2f}"
            else:
                txt = f"P{i+1}"
            
            ax.text(cx + 6, cy, txt, color='white', fontsize=7, fontweight='bold', 
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    # --- 8. 开始绘图 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # Row 1
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB: {meta['scene']}\nPitch:{meta['pitch']:.1f}", fontsize=10)
    axes[0, 0].axis('off')
    draw_probes(axes[0, 0], rgb, PROBE_POINTS, 'rgb')

    im2 = axes[0, 1].imshow(gt, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"GT (Clip: {vmin:.1f}-{vmax:.1f}m)", fontsize=10)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    draw_probes(axes[0, 1], gt, PROBE_POINTS, 'depth')

    im3 = axes[0, 2].imshow(pred_aligned, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"Pred (Aligned)\nScale: {scale_shift:.2f}", fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    draw_probes(axes[0, 2], pred_aligned, PROBE_POINTS, 'depth')

    # Row 2
    im4 = axes[1, 0].imshow(abs_diff, cmap='turbo', vmin=0, vmax=5.0) 
    axes[1, 0].set_title("Abs Error (Red > 5m)", fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(precision_map, cmap=cmap_prec, norm=norm_prec, interpolation='nearest')
    axes[1, 1].set_title("Precision Metrics", fontsize=10)
    axes[1, 1].axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00cc00', label=r'Elite (<1.05)'),
        Patch(facecolor='#00ffff', label=r'High (<1.15)'),
        Patch(facecolor='#ffdb14', label=r'Std (<1.25)'),
        Patch(facecolor='#999999', label=r'Noise (<1m)'),
        Patch(facecolor='#ff0000', label=r'Fail (>1.25)'),
    ]
    axes[1, 1].legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.7)
    draw_probes(axes[1, 1], thresh, PROBE_POINTS, 'ratio')

    limit = min(max(abs(np.nanmin(log_diff)), abs(np.nanmax(log_diff))), 1.0)
    im6 = axes[1, 2].imshow(log_diff, cmap='seismic', vmin=-limit, vmax=limit)
    axes[1, 2].set_title("Structure Diff", fontsize=10)
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Scene: {meta['scene']} | Img: {os.path.basename(save_path)}", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# class CSVIndexDataset(Dataset):
#     def __init__(self, csv_path, pred_root, gt_roots):
#         self.df = pd.read_csv(csv_path)
#         self.val_df = self.df[self.df['Split'] == 'val'].reset_index(drop=True)
#         self.pred_root = Path(pred_root)
#         self.gt_roots = [Path(p) for p in gt_roots]
        
#         # 缓存一下目录结构，避免每次都 listdir (加速 IO)
#         self.dir_cache = {} 
        
#         print(f"Dataset initialized. Val samples: {len(self.val_df)}")

#     def __len__(self): return len(self.val_df)
    
#     def _get_dir_files(self, dir_path):
#         """缓存目录下的文件名，加速搜索"""
#         s_path = str(dir_path)
#         if s_path not in self.dir_cache:
#             if dir_path.exists():
#                 self.dir_cache[s_path] = os.listdir(s_path)
#             else:
#                 self.dir_cache[s_path] = []
#         return self.dir_cache[s_path]

#     def find_file_smart(self, roots, scene, sub, stems, exts=['.npy']):
#         """
#         智能查找文件：
#         1. 精确匹配
#         2. 忽略大小写匹配
#         3. 前缀匹配/包含匹配 (解决时间戳截断问题)
#         """
#         for root in roots:
#             base = root / scene / sub
#             if not base.exists(): continue
            
#             # 1. 尝试直接拼接路径 (最快)
#             for stem in stems:
#                 for ext in exts:
#                     cand = base / f"{stem}{ext}"
#                     if cand.exists(): return cand

#             # 2. 既然找不到，就获取该目录下所有文件进行“模糊搜索”
#             # 获取该目录下所有文件名 (带缓存)
#             all_files = self._get_dir_files(base)
#             if not all_files: continue

#             # 构建查找映射 (转小写)
#             lower_map = {f.lower(): f for f in all_files}
            
#             for stem in stems:
#                 clean_stem = stem.strip()
                
#                 # 策略 A: 忽略大小写精确匹配
#                 for ext in exts:
#                     target_name = f"{clean_stem}{ext}".lower()
#                     if target_name in lower_map:
#                         return base / lower_map[target_name]

#                 # 策略 B: 模糊匹配 (针对时间戳问题)
#                 # 如果 stem 是 "1698217951.599842072"，而文件是 "1698217951.599.npy"
#                 # 我们检查 startswith 或 contains
#                 for f_name in all_files:
#                     f_lower = f_name.lower()
#                     stem_lower = clean_stem.lower()
                    
#                     # 检查 1: 文件名包含了 stem (文件比 stem 长)
#                     if stem_lower in f_lower:
#                         # 确保扩展名匹配
#                         if any(f_lower.endswith(e) for e in exts):
#                             return base / f_name
                            
#                     # 检查 2: stem 包含了文件名 (stem 比 文件 长 - 针对截断)
#                     # 去掉扩展名比较
#                     f_stem = os.path.splitext(f_lower)[0]
#                     if f_stem in stem_lower and len(f_stem) > 10: # 长度限制防止匹配到 "1.npy" 这种短的
#                          if any(f_lower.endswith(e) for e in exts):
#                             return base / f_name
                            
#         return None

#     def __getitem__(self, idx):
#         row = self.val_df.iloc[idx]
#         scene = str(row['Scene_Name'])
#         stems = []
#         if pd.notna(row.get('Renamed_Image')): stems.append(os.path.splitext(str(row['Renamed_Image']))[0])
#         if pd.notna(row.get('Original_Filename')): stems.append(os.path.splitext(str(row['Original_Filename']))[0])
        
#         # 使用智能搜索
#         gt_path = self.find_file_smart(self.gt_roots, scene, "depth", stems, ['.npy'])
#         pred_path = self.find_file_smart([self.pred_root], scene, "", stems, ['.npy'])
        
#         # 记录缺失日志
#         if gt_path is None or pred_path is None:
#             with open(MISSING_LOG_PATH, "a") as f:
#                 missing_type = []
#                 if gt_path is None: missing_type.append("GT")
#                 if pred_path is None: missing_type.append("PRED")
#                 f.write(f"[Idx {idx}] Scene: {scene} | Missing: {', '.join(missing_type)} | Stems: {stems}\n")
#             return None 
            
#         try:
#             gt = np.load(gt_path).astype(np.float32)
#             pred = np.load(pred_path).astype(np.float32)
            
#             # 查找 RGB (支持多种文件夹名)
#             rgb_path = None
#             for sub in ["images", "rgb", "img", ""]:
#                 rgb_path = self.find_file_smart(self.gt_roots, scene, sub, stems, ['.jpg', '.png', '.jpeg', '.bmp'])
#                 if rgb_path: break
            
#             # 使用安全加载函数 (修复维度报错的关键！)
#             target_hw = gt.shape[:2]
#             rgb = safe_load_rgb(rgb_path, target_hw=target_hw)

#             focal_val = row.get('FocalLength_New', row.get('FocalLength(px)', 0))
#             meta = {
#                 'scene': scene,
#                 'pitch': row.get('Visual_Pitch(deg)', 0),
#                 'focal': focal_val,
#                 'idx': idx
#             }
            
#             # 此时 rgb 保证是 (H, W, 3)，可以安全 permute
#             return torch.from_numpy(pred), torch.from_numpy(gt), torch.from_numpy(rgb).permute(2,0,1), meta
            
#         except Exception as e:
#             with open(MISSING_LOG_PATH, "a") as f:
#                 f.write(f"[Idx {idx}] Scene: {scene} | Load Error: {e}\n")
#             return None

class CSVIndexDataset(Dataset):
    def __init__(self, csv_path, pred_root, gt_roots):
        self.df = pd.read_csv(csv_path)
        self.val_df = self.df[self.df['Split'] == 'val'].reset_index(drop=True)
        self.pred_root = Path(pred_root)
        self.gt_roots = [Path(p) for p in gt_roots]
        
        # 缓存目录结构
        self.dir_cache = {} 
        print(f"Dataset initialized. Val samples: {len(self.val_df)}")

    def __len__(self): return len(self.val_df)
    
    def _get_dir_files(self, dir_path):
        s_path = str(dir_path)
        if s_path not in self.dir_cache:
            if dir_path.exists():
                self.dir_cache[s_path] = os.listdir(s_path)
            else:
                self.dir_cache[s_path] = []
        return self.dir_cache[s_path]

    def find_file_smart(self, roots, scene, sub, stems, exts=['.npy']):
        for root in roots:
            base = root / scene / sub
            if not base.exists(): continue
            
            # 1. 快速精确匹配
            for stem in stems:
                for ext in exts:
                    cand = base / f"{stem}{ext}"
                    if cand.exists(): return cand

            # 2. 模糊/不区分大小写搜索
            all_files = self._get_dir_files(base)
            if not all_files: continue
            lower_map = {f.lower(): f for f in all_files}
            
            for stem in stems:
                clean_stem = stem.strip().lower()
                
                # 策略 A: 精确匹配 (忽略大小写)
                for ext in exts:
                    target = f"{clean_stem}{ext.lower()}"
                    if target in lower_map: return base / lower_map[target]

                # 策略 B: 包含匹配 (针对时间戳)
                for f_name in all_files:
                    f_lower = f_name.lower()
                    # Stem 包含文件名 (如 stems=['123.456'], file='123.jpg')
                    # 或 文件名包含 Stem (如 stems=['123'], file='123_view.jpg')
                    # 且后缀匹配
                    if any(f_lower.endswith(e.lower()) for e in exts):
                        f_stem = os.path.splitext(f_lower)[0]
                        if (clean_stem in f_stem) or (f_stem in clean_stem and len(f_stem) > 5):
                             return base / f_name
        return None

    def __getitem__(self, idx):
        row = self.val_df.iloc[idx]
        scene = str(row['Scene_Name'])
        stems = []
        if pd.notna(row.get('Renamed_Image')): stems.append(os.path.splitext(str(row['Renamed_Image']))[0])
        if pd.notna(row.get('Original_Filename')): stems.append(os.path.splitext(str(row['Original_Filename']))[0])
        
        # 1. 查找 GT 和 Pred
        gt_path = self.find_file_smart(self.gt_roots, scene, "depth", stems, ['.npy'])
        pred_path = self.find_file_smart([self.pred_root], scene, "", stems, ['.npy'])
        
        if gt_path is None or pred_path is None:
            # 记录缺失日志... (代码略，保持你原有的逻辑)
            return None 
            
        try:
            gt = np.load(gt_path).astype(np.float32)
            pred = np.load(pred_path).astype(np.float32)
            
            # 2. 查找 RGB (增强搜索逻辑)
            # 增加 .JPG, .TIF 等大写后缀，以及 .tiff
            img_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']
            rgb_path = None
            
            # 尝试多个子文件夹名
            for sub in ["rgbs","images", "rgb", "img", "imgs", ""]:
                rgb_path = self.find_file_smart(self.gt_roots, scene, sub, stems, img_exts)
                if rgb_path: break
            
            # 3. 读取 RGB (确保使用 safe_load_rgb)
            target_hw = gt.shape[:2]
            # 注意：这里需要调用全局函数 safe_load_rgb
            rgb = safe_load_rgb(rgb_path, target_hw=target_hw)

            # 如果没找到 RGB，记录一条 Warning 日志 (方便排查)
            if rgb_path is None:
                 with open(MISSING_LOG_PATH, "a") as f:
                    f.write(f"[Idx {idx}] Scene: {scene} | Warning: RGB Not Found | Stems: {stems}\n")

            focal_val = row.get('FocalLength_New', row.get('FocalLength(px)', 0))
            meta = {
                'scene': scene,
                'pitch': row.get('Visual_Pitch(deg)', 0),
                'focal': focal_val,
                'idx': idx
            }
            
            return torch.from_numpy(pred), torch.from_numpy(gt), torch.from_numpy(rgb).permute(2,0,1), meta
            
        except Exception as e:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]

# ================= 辅助函数 =================
def format_line(name, m, count_img=0, count_scn=0, indent=0):
    sp = " " * indent
    m = np.nan_to_num(m)
    return "{:<20} | {:>4} | {:>4} | {:>6.4f} | {:>6.2f} | {:>6.4f} | {:>6.4f} | {:>6.4f} | {:>6.4f} || {:>6.4f} | {:>6.4f} | {:>6.4f} |".format(
        sp + str(name)[-20+indent:], count_img, count_scn, m[0], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9])

def format_header():
    return "{:<20} | {:>4} | {:>4} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} || {:>6} | {:>6} | {:>6} |".format(
        "Category", "Img#", "Scn#", "AbsRel", "RMSE", "RMSElg", "a1", "a2", "a3", "SI-Log", "Spear", "N-RMSE")

# ================= 主流程 =================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CSVIndexDataset(MASTER_CSV, PRED_ROOT, VAL_GT_ROOTS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    acc_overall = torch.zeros(len(ALL_LABELS), 12).to(device)
    acc_pitch, acc_focal, acc_scene, acc_category = {}, {}, {}, {}
    count_pitch, count_focal, count_scene, count_category = {}, {}, {}, {}
    total_imgs = 0
    
    print(f"🚀 开始全量评估 + 可视化 (Interval={args.viz_interval})...")
    print(f"📂 可视化结果将保存至: {VIZ_DIR}")
    print(f"📄 缺失文件日志将保存至: {MISSING_LOG_PATH}")
    
    for batch in tqdm(loader):
        if batch is None: continue
        preds, gts, rgbs, metas = batch
        
        for i in range(len(preds)):
            pred, gt, rgb, meta = preds[i].to(device), gts[i].to(device), rgbs[i], metas[i]
            
            if pred.dim() == 3: pred = pred.squeeze(-1)
            if gt.dim() == 3: gt = gt.squeeze(-1)
            if pred.shape != gt.shape:
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear').squeeze()
            
            mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH)
            if mask.sum() < 10: continue
            
            if USE_MEDIAN_SCALING: pred = align_scale_shift_torch(pred, gt, mask)
            pred.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)
            
            batch_sums = compute_errors_torch_bins(gt.unsqueeze(0), pred.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            
            total_imgs += 1
            if total_imgs % args.viz_interval == 0:
                viz_name = f"{meta['scene']}_{meta['idx']}.png"
                visualize_sample(rgb, gt.cpu(), pred.cpu(), mask.cpu(), os.path.join(VIZ_DIR, viz_name), meta)
            
            acc_overall += batch_sums
            
            def update_stats(acc_dict, count_dict, key, vals, scene_name):
                if key not in acc_dict:
                    acc_dict[key] = torch.zeros_like(vals)
                    count_dict[key] = {'img': 0, 'scn': set()}
                acc_dict[key] += vals
                count_dict[key]['img'] += 1
                count_dict[key]['scn'].add(scene_name)

            p_val = float(meta['pitch'])
            p_key = "-90to-75" if p_val < -75 else "-75to-60" if p_val < -60 else "-60to-45" if p_val < -45 else "-45to-30" if p_val < -30 else "-30to-0"
            update_stats(acc_pitch, count_pitch, p_key, batch_sums, meta['scene'])
            
            f_val = meta['focal']
            f_start = int(f_val // FOCAL_STEP) * FOCAL_STEP
            update_stats(acc_focal, count_focal, f"{f_start}-{f_start + FOCAL_STEP}", batch_sums, meta['scene'])
            
            update_stats(acc_scene, count_scene, meta['scene'], batch_sums, meta['scene'])
            update_stats(acc_category, count_category, get_scene_category(meta['scene']), batch_sums, meta['scene'])

    # 报告生成 (与之前相同，节省篇幅)
    lines = []
    header = format_header()
    sep = "-" * len(header)
    lines += ["EVALUATION REPORT", f"Date: {datetime.now()}", "="*130, header, sep]
    
    ov_res = compute_metrics_from_sums(acc_overall[-1].cpu().numpy())
    all_scenes = set()
    for v in count_scene.values(): all_scenes.update(v['scn'])
    lines.append(format_line("OVERALL", ov_res, total_imgs, len(all_scenes)))
    lines.append(sep)
    
    lines.append(">>> BY SCENE CATEGORY")
    for k in [c for c in ["Rural", "Natural", "City", "Factory", "Uncategorized"] if c in acc_category]:
        lines.append(format_line(k, compute_metrics_from_sums(acc_category[k][-1].cpu().numpy()), count_category[k]['img'], len(count_category[k]['scn'])))
    lines.append(sep)

    lines.append(">>> ALL SCENES (Sorted by Spearman)")
    scene_list = sorted([(k, compute_metrics_from_sums(acc_scene[k][-1].cpu().numpy()), count_scene[k]) for k in acc_scene], key=lambda x: x[1][8], reverse=True)
    for k, res, cnt in scene_list:
        lines.append(format_line(k, res, cnt['img'], len(cnt['scn'])))
    
    report_text = "\n".join(lines)
    print(report_text)
    with open(os.path.join(PRED_ROOT, "Final_Report_Viz.txt"), "w") as f: f.write(report_text)
    print(f"\n✅ 报告已保存至: {os.path.join(PRED_ROOT, 'Final_Report_Viz.txt')}")

'''
python /home/szq/moge2/MoGe/moge/scripts/c-real-tasks-pitch-altitude-fov-syn-debug-batch-classes4-vis.py \
  --pred_root /data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k-out \
  --gt_root /data1/szq/data/Val \
  --csv_path /home/szq/moge2/final_merged.csv \
  --viz_interval 20

'''