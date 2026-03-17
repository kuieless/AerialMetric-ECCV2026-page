# # import os
# # import cv2
# # import argparse
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# # from pathlib import Path
# # from tqdm import tqdm
# # import warnings

# # # ================= ⚙️ 配置区域 =================
# # # 全局深度单位换算: PNG数值 / 1000.0 = 米
# # GLOBAL_SCALE = 1000.0 
# # # 无效值阈值: 大于 65000 (如 65535) 视为无效
# # INVALID_DEPTH_THRESH = 65000 

# # # 评估有效范围 (米)
# # MIN_EVAL_DEPTH = 1e-3
# # MAX_EVAL_DEPTH = 80.0 

# # # 是否启用中值缩放 (对于 Zero-shot 评估通常设为 True)
# # USE_MEDIAN_SCALING = True 

# # # DataLoader 配置
# # NUM_WORKERS = 8
# # BATCH_SIZE = 1
# # # ==============================================

# # def compute_errors(gt, pred):
# #     """
# #     计算标准深度估计指标
# #     输入:
# #         gt:   Ground Truth Tensor (N)
# #         pred: Prediction Tensor (N)
# #     输出:
# #         Tensor [AbsRel, SqRel, RMSE, RMSElog, a1, a2, a3, count]
# #     """
# #     thresh = torch.max((gt / pred), (pred / gt))
# #     a1 = (thresh < 1.25     ).float().mean()
# #     a2 = (thresh < 1.25 ** 2).float().mean()
# #     a3 = (thresh < 1.25 ** 3).float().mean()

# #     rmse = (gt - pred) ** 2
# #     rmse = torch.sqrt(rmse.mean())

# #     rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
# #     rmse_log = torch.sqrt(rmse_log.mean())

# #     abs_rel = torch.mean(torch.abs(gt - pred) / gt)
# #     sq_rel = torch.mean(((gt - pred) ** 2) / gt)

# #     # 返回累加需要的数值 (注意：这里返回的是均值，但在聚合时我们会用 count 加权还原回 sum)
# #     # 格式: [AbsRel, SqRel, RMSE, RMSElog, a1, a2, a3, PixelCount]
# #     return torch.tensor([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, gt.numel()], device=gt.device)

# # def align_scale_shift_torch(pred, target, mask):
# #     """
# #     使用中值对齐 (Median Scaling)
# #     """
# #     # 仅在有效区域计算中值
# #     safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    
# #     if safe_mask.sum() == 0:
# #         return pred # 无法对齐，保持原样
        
# #     t_median = torch.median(target[safe_mask])
# #     p_median = torch.median(pred[safe_mask])
    
# #     ratio = t_median / (p_median + 1e-8)
# #     return pred * ratio

# # class BenchmarkDataset(Dataset):
# #     def __init__(self, csv_path, pred_root):
# #         self.df = pd.read_csv(csv_path)
# #         self.pred_root = Path(pred_root)
# #         print(f"📊 Dataset Loaded. Total Samples: {len(self.df)}")
# #         print(f"📂 Prediction Root: {self.pred_root}")

# #     def __len__(self):
# #         return len(self.df)
    
# #     def load_gt(self, path):
# #         if not os.path.exists(path):
# #             return None
            
# #         # 读取原始 16-bit PNG
# #         # cv2.IMREAD_UNCHANGED 非常重要，否则会被转成 8-bit
# #         depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
# #         if depth_raw is None:
# #             return None
            
# #         depth = depth_raw.astype(np.float32)
        
# #         # 1. 处理无效像素 (Sky / Infinity)
# #         # 将 65535 等大值设为 0，后续 Mask 会忽略 0 值
# #         depth[depth > INVALID_DEPTH_THRESH] = 0.0
        
# #         # 2. 单位换算 (mm -> m)
# #         depth = depth / GLOBAL_SCALE
        
# #         return depth

# #     def __getitem__(self, idx):
# #         row = self.df.iloc[idx]
        
# #         # 获取信息
# #         dataset_name = row['Dataset']
# #         renamed_image = row['Renamed_Image'] # e.g., "DIODE_scene01_000.jpg"
# #         gt_path = row['GT_Path']
        
# #         # 寻找对应的推理结果 (.npy)
# #         # Step 1/2 可能生成 "DIODE_scene01_000.npy" 或 "DIODE_scene01_000.jpg.npy"
# #         # 这里的逻辑是尝试寻找文件
        
# #         stem_name = renamed_image
# #         candidates = [
# #             self.pred_root / (stem_name + ".npy"),           # Case A: image.jpg.npy
# #             self.pred_root / (os.path.splitext(stem_name)[0] + ".npy") # Case B: image.npy
# #         ]
        
# #         pred_path = None
# #         for p in candidates:
# #             if p.exists():
# #                 pred_path = p
# #                 break
        
# #         if pred_path is None:
# #             # 如果找不到推理文件，返回 None，在 collate_fn 中跳过
# #             return None
        
# #         # 加载数据
# #         try:
# #             pred = np.load(pred_path) # 假设 pred 已经是 float32 (meters) 或 disparity
# #             gt = self.load_gt(gt_path)
# #         except Exception as e:
# #             print(f"Error loading {renamed_image}: {e}")
# #             return None

# #         if gt is None:
# #             return None
            
# #         # 转换为 Tensor
# #         pred_tensor = torch.from_numpy(pred).float()
# #         gt_tensor = torch.from_numpy(gt).float()
        
# #         meta = {
# #             'dataset': dataset_name,
# #             'scene': renamed_image,
# #             'gt_path': gt_path
# #         }
        
# #         return pred_tensor, gt_tensor, meta

# # def collate_fn(batch):
# #     # 过滤掉 None (即找不到文件的情况)
# #     batch = [b for b in batch if b is not None]
# #     if len(batch) == 0:
# #         return None
# #     preds, gts, metas = zip(*batch)
# #     return preds, gts, metas

# # def main():
# #     parser = argparse.ArgumentParser(description="Full Benchmark Evaluation Step 3")
# #     parser.add_argument("--pred_root", type=str, required=True, help="Step 2 输出的 NPY 文件夹")
# #     parser.add_argument("--csv_path", type=str, required=True, help="Step 0 生成的 CSV 索引")
# #     parser.add_argument("--save_report", type=str, default="benchmark_report.txt", help="保存结果的文件路径")
# #     args = parser.parse_args()

# #     # 设备配置
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# #     # 数据集准备
# #     dataset = BenchmarkDataset(args.csv_path, args.pred_root)
# #     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
# #                         num_workers=NUM_WORKERS, collate_fn=collate_fn)

# #     # 统计累加器
# #     # 结构: { 'DIODE': {'abs_rel_sum': 0.0, 'pixel_count': 0, ...}, 'KITTI': ... }
# #     stats_accumulator = {}

# #     print(f"🚀 开始评估... Device: {device}")
    
# #     for batch in tqdm(loader, desc="Evaluating"):
# #         if batch is None: continue
        
# #         preds, gts, metas = batch
        
# #         for i in range(len(preds)):
# #             pred = preds[i].to(device)
# #             gt = gts[i].to(device)
# #             d_name = metas[i]['dataset']
            
# #             # 1. 尺寸对齐 (如果 Pred 和 GT 尺寸不一致，插值 Pred)
# #             if pred.shape != gt.shape:
# #                 # pred 需要是 (1, 1, H, W)
# #                 pred = F.interpolate(
# #                     pred.unsqueeze(0).unsqueeze(0), 
# #                     size=gt.shape[-2:], 
# #                     mode='bilinear', 
# #                     align_corners=False
# #                 ).squeeze()
            
# #             # 2. 生成有效掩码 Mask
# #             # 规则: GT > min, GT < max, GT != 0 (处理之前的 invalid)
# #             mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH)
            
# #             # 如果有效像素太少，跳过
# #             if mask.sum() < 10:
# #                 continue
                
# #             # 3. 对齐 (Median Scaling)
# #             if USE_MEDIAN_SCALING:
# #                 pred = align_scale_shift_torch(pred, gt, mask)
            
# #             # 4. 计算指标
# #             # 仅取 mask 部分计算
# #             pred_valid = pred[mask]
# #             gt_valid = gt[mask]
            
# #             # 防止 pred 出现负数或0 (log误差会报错)
# #             pred_valid = torch.clamp(pred_valid, min=1e-3)
            
# #             metrics = compute_errors(gt_valid, pred_valid)
# #             # metrics format: [AbsRel, SqRel, RMSE, RMSElog, a1, a2, a3, Count]
            
# #             # 5. 累加统计
# #             if d_name not in stats_accumulator:
# #                 stats_accumulator[d_name] = {
# #                     'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse_sum_sq': 0.0, 'rmse_log_sum_sq': 0.0,
# #                     'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'count': 0
# #                 }
            
# #             # 注意: RMSE 需要累加平方和，最后开根号；AbsRel 需要累加 sum
# #             # compute_errors 返回的是 mean，所以要乘以 count 还原为 sum
# #             pixel_count = metrics[7].item()
            
# #             stats_accumulator[d_name]['abs_rel'] += metrics[0].item() * pixel_count
# #             stats_accumulator[d_name]['sq_rel']  += metrics[1].item() * pixel_count
# #             stats_accumulator[d_name]['rmse_sum_sq'] += (metrics[2].item() ** 2) * pixel_count
# #             stats_accumulator[d_name]['rmse_log_sum_sq'] += (metrics[3].item() ** 2) * pixel_count
# #             stats_accumulator[d_name]['a1'] += metrics[4].item() * pixel_count
# #             stats_accumulator[d_name]['a2'] += metrics[5].item() * pixel_count
# #             stats_accumulator[d_name]['a3'] += metrics[6].item() * pixel_count
# #             stats_accumulator[d_name]['count'] += pixel_count

# #     # ================= 📊 生成报告 =================
# #     print("\n" + "="*100)
# #     header = f"{'Dataset':<15} | {'AbsRel':<8} | {'SqRel':<8} | {'RMSE':<8} | {'RMSElog':<8} | {'a1':<8} | {'a2':<8} | {'a3':<8}"
# #     print(header)
# #     print("-" * 100)
    
# #     report_lines = [header, "-" * 100]

# #     for d_name in sorted(stats_accumulator.keys()):
# #         data = stats_accumulator[d_name]
# #         count = data['count']
        
# #         if count > 0:
# #             m_abs_rel = data['abs_rel'] / count
# #             m_sq_rel  = data['sq_rel'] / count
# #             m_rmse    = np.sqrt(data['rmse_sum_sq'] / count)
# #             m_rmse_log= np.sqrt(data['rmse_log_sum_sq'] / count)
# #             m_a1      = data['a1'] / count
# #             m_a2      = data['a2'] / count
# #             m_a3      = data['a3'] / count
            
# #             line = f"{d_name:<15} | {m_abs_rel:.4f}   | {m_sq_rel:.4f}   | {m_rmse:.4f}   | {m_rmse_log:.4f}   | {m_a1:.3f}    | {m_a2:.3f}    | {m_a3:.3f}"
# #             print(line)
# #             report_lines.append(line)
# #         else:
# #             print(f"{d_name:<15} | No valid pixels found")

# #     print("="*100)
    
# #     # 保存结果到文件
# #     with open(args.save_report, 'w') as f:
# #         f.write('\n'.join(report_lines))
# #     print(f"📄 Report saved to {args.save_report}")

# # if __name__ == "__main__":
# #     main()
# import os
# import cv2
# import argparse
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
# from tqdm import tqdm
# import warnings

# # ================= ⚙️ 配置区域 =================
# # 全局深度单位换算: PNG数值 / 1000.0 = 米
# GLOBAL_SCALE = 1000.0 
# # 无效值阈值: 大于 60000 视为无效 (通常 65535 是无效值)
# INVALID_DEPTH_THRESH = 60000 

# # 评估有效范围 (米) - 根据之前的分析，这些数据集通常在 80m 或 100m 以内
# MIN_EVAL_DEPTH = 1e-3
# MAX_EVAL_DEPTH = 100.0 

# # 是否启用中值缩放 (Scale Alignment)
# # MoGe 输出是相对深度 (metric_scale 只有在特定模式下才准)，跨数据集评估通常需要对齐
# USE_MEDIAN_SCALING = True 

# # DataLoader 配置
# NUM_WORKERS = 8
# BATCH_SIZE = 1
# # ==============================================

# def compute_errors(gt, pred):
#     """
#     计算标准深度估计指标
#     输入:
#         gt:   Ground Truth Tensor (N)
#         pred: Prediction Tensor (N)
#     输出:
#         Tensor [AbsRel, SqRel, RMSE, RMSElog, a1, a2, a3, count]
#     """
#     thresh = torch.max((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25     ).float().mean()
#     a2 = (thresh < 1.25 ** 2).float().mean()
#     a3 = (thresh < 1.25 ** 3).float().mean()

#     rmse = (gt - pred) ** 2
#     rmse = torch.sqrt(rmse.mean())

#     rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
#     rmse_log = torch.sqrt(rmse_log.mean())

#     abs_rel = torch.mean(torch.abs(gt - pred) / gt)
#     sq_rel = torch.mean(((gt - pred) ** 2) / gt)

#     # 返回累加需要的数值
#     return torch.tensor([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, gt.numel()], device=gt.device)

# def align_scale_shift_torch(pred, target, mask):
#     """
#     使用中值对齐 (Median Scaling)
#     s = median(gt) / median(pred)
#     """
#     safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    
#     if safe_mask.sum() == 0:
#         return pred 
        
#     t_median = torch.median(target[safe_mask])
#     p_median = torch.median(pred[safe_mask])
    
#     if p_median < 1e-6:
#         return pred

#     ratio = t_median / p_median
#     return pred * ratio

# class BenchmarkDataset(Dataset):
#     def __init__(self, csv_path, pred_root):
#         self.df = pd.read_csv(csv_path)
#         self.pred_root = Path(pred_root)
#         print(f"📊 Dataset Loaded. Total Samples: {len(self.df)}")
#         print(f"📂 Prediction Root: {self.pred_root}")

#     def __len__(self):
#         return len(self.df)
    
#     def load_gt(self, path):
#         if not os.path.exists(path):
#             return None
            
#         # 读取原始 16-bit PNG
#         # cv2.IMREAD_UNCHANGED (-1) 非常重要
#         depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
#         if depth_raw is None:
#             return None
            
#         depth = depth_raw.astype(np.float32)
        
#         # 1. 处理无效像素
#         depth[depth > INVALID_DEPTH_THRESH] = 0.0
        
#         # 2. 单位换算 (mm -> m)
#         depth = depth / GLOBAL_SCALE
        
#         return depth

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
        
#         # 获取信息
#         dataset_name = row['Dataset']
#         renamed_image = row['Renamed_Image'] # e.g., "DIODE_scene01_000.jpg"
#         gt_path = row['GT_Path']
        
#         # 寻找对应的推理结果 (.npy)
#         # Step 2 生成的文件名是: Renamed_Image (去掉后缀) + .npy
#         # 例如: DIODE_scene01_000.jpg -> DIODE_scene01_000.npy
        
#         stem_name = os.path.splitext(renamed_image)[0]
#         pred_path = self.pred_root / (stem_name + ".npy")
        
#         if not pred_path.exists():
#             # 兼容性尝试: 有些步骤可能保留了 .jpg 后缀
#             pred_path_alt = self.pred_root / (renamed_image + ".npy")
#             if pred_path_alt.exists():
#                 pred_path = pred_path_alt
#             else:
#                 return None # 找不到预测文件，跳过
        
#         # 加载数据
#         try:
#             pred = np.load(pred_path) 
#             gt = self.load_gt(gt_path)
#         except Exception as e:
#             print(f"Error loading {renamed_image}: {e}")
#             return None

#         if gt is None:
#             return None
            
#         # 转换为 Tensor
#         pred_tensor = torch.from_numpy(pred).float()
#         gt_tensor = torch.from_numpy(gt).float()
        
#         meta = {
#             'dataset': dataset_name,
#             'scene': renamed_image,
#         }
        
#         return pred_tensor, gt_tensor, meta

# def collate_fn(batch):
#     batch = [b for b in batch if b is not None]
#     if len(batch) == 0:
#         return None
#     preds, gts, metas = zip(*batch)
#     return preds, gts, metas

# def main():
#     parser = argparse.ArgumentParser(description="Full Benchmark Evaluation Step 3")
#     parser.add_argument("--pred_root", type=str, required=True, help="Step 2 输出的 NPY 文件夹")
#     parser.add_argument("--csv_path", type=str, required=True, help="Step 0 生成的 CSV 索引")
#     parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")
#     args = parser.parse_args()

#     # 1. 准备环境
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)
#     report_path = os.path.join(args.output_dir, "Final_Report.txt")
    
#     # 2. 加载数据
#     dataset = BenchmarkDataset(args.csv_path, args.pred_root)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
#                         num_workers=NUM_WORKERS, collate_fn=collate_fn)

#     # 3. 统计累加器
#     # 结构: { 'DIODE': {'abs_rel': 0.0, ...}, 'KITTI': ... }
#     stats = {}

#     print(f"🚀 [Step 3] 开始评估... (Median Scaling: {USE_MEDIAN_SCALING})")
    
#     for batch in tqdm(loader, desc="Evaluating"):
#         if batch is None: continue
        
#         preds, gts, metas = batch
        
#         for i in range(len(preds)):
#             pred = preds[i].to(device)
#             gt = gts[i].to(device)
#             d_name = metas[i]['dataset']
            
#             # A. 尺寸对齐 (Pred -> GT size)
#             if pred.shape != gt.shape:
#                 pred = F.interpolate(
#                     pred.unsqueeze(0).unsqueeze(0), 
#                     size=gt.shape[-2:], 
#                     mode='bilinear', 
#                     align_corners=False
#                 ).squeeze()
            
#             # B. 生成有效掩码 Mask
#             # 规则: GT > min, GT < max
#             mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH)
            
#             if mask.sum() < 10: continue
                
#             # C. 对齐 (Median Scaling)
#             if USE_MEDIAN_SCALING:
#                 pred = align_scale_shift_torch(pred, gt, mask)
            
#             # D. 计算指标 (仅在 Mask 区域)
#             pred_valid = pred[mask]
#             gt_valid = gt[mask]
            
#             # 防止 log 报错
#             pred_valid = torch.clamp(pred_valid, min=1e-3)
            
#             metrics = compute_errors(gt_valid, pred_valid)
#             # metrics: [AbsRel, SqRel, RMSE, RMSElog, a1, a2, a3, Count]
            
#             # E. 累加到对应数据集
#             if d_name not in stats:
#                 stats[d_name] = {
#                     'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse_sum_sq': 0.0, 'rmse_log_sum_sq': 0.0,
#                     'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'count': 0
#                 }
            
#             pixel_count = metrics[7].item()
            
#             # 累加：平均指标 * 像素数 = 总误差和
#             stats[d_name]['abs_rel'] += metrics[0].item() * pixel_count
#             stats[d_name]['sq_rel']  += metrics[1].item() * pixel_count
#             # RMSE 需要累加平方和，最后除以总数再开根
#             stats[d_name]['rmse_sum_sq'] += (metrics[2].item() ** 2) * pixel_count
#             stats[d_name]['rmse_log_sum_sq'] += (metrics[3].item() ** 2) * pixel_count
            
#             stats[d_name]['a1'] += metrics[4].item() * pixel_count
#             stats[d_name]['a2'] += metrics[5].item() * pixel_count
#             stats[d_name]['a3'] += metrics[6].item() * pixel_count
            
#             stats[d_name]['count'] += pixel_count

#     # ================= 📊 生成报告 =================
#     print("\n" + "="*110)
#     header = f"{'Dataset':<15} | {'Count':<8} | {'AbsRel':<8} | {'SqRel':<8} | {'RMSE':<8} | {'RMSElog':<8} | {'a1':<8} | {'a2':<8} | {'a3':<8}"
#     print(header)
#     print("-" * 110)
    
#     report_lines = [f"Evaluation Report (Median Scaling={USE_MEDIAN_SCALING})", "="*110, header, "-" * 110]
    
#     # 计算 Overall
#     overall_stats = {k: 0.0 for k in stats[list(stats.keys())[0]].keys()}

#     for d_name in sorted(stats.keys()):
#         data = stats[d_name]
#         count = data['count']
        
#         # 累加到 Overall
#         for k in overall_stats:
#             overall_stats[k] += data[k]
        
#         if count > 0:
#             m_abs_rel = data['abs_rel'] / count
#             m_sq_rel  = data['sq_rel'] / count
#             m_rmse    = np.sqrt(data['rmse_sum_sq'] / count)
#             m_rmse_log= np.sqrt(data['rmse_log_sum_sq'] / count)
#             m_a1      = data['a1'] / count
#             m_a2      = data['a2'] / count
#             m_a3      = data['a3'] / count
            
#             # 图片数量（估算）
#             img_count = len(dataset.df[dataset.df['Dataset'] == d_name])
            
#             line = f"{d_name:<15} | {img_count:<8} | {m_abs_rel:.4f}   | {m_sq_rel:.4f}   | {m_rmse:.4f}   | {m_rmse_log:.4f}   | {m_a1:.3f}    | {m_a2:.3f}    | {m_a3:.3f}"
#             print(line)
#             report_lines.append(line)

#     print("-" * 110)
    
#     # 计算 Overall 均值
#     total_count = overall_stats['count']
#     if total_count > 0:
#         o_abs_rel = overall_stats['abs_rel'] / total_count
#         o_sq_rel  = overall_stats['sq_rel'] / total_count
#         o_rmse    = np.sqrt(overall_stats['rmse_sum_sq'] / total_count)
#         o_rmse_log= np.sqrt(overall_stats['rmse_log_sum_sq'] / total_count)
#         o_a1      = overall_stats['a1'] / total_count
#         o_a2      = overall_stats['a2'] / total_count
#         o_a3      = overall_stats['a3'] / total_count
        
#         line = f"{'OVERALL':<15} | {len(dataset):<8} | {o_abs_rel:.4f}   | {o_sq_rel:.4f}   | {o_rmse:.4f}   | {o_rmse_log:.4f}   | {o_a1:.3f}    | {o_a2:.3f}    | {o_a3:.3f}"
#         print(line)
#         report_lines.append("-" * 110)
#         report_lines.append(line)

#     print("="*110)
    
#     with open(report_path, 'w') as f:
#         f.write('\n'.join(report_lines))
#     print(f"📄 Report saved to {report_path}")

# if __name__ == "__main__":
#     main()
import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import warnings

# ================= ⚙️ 配置区域 =================
GLOBAL_SCALE = 1000.0 
INVALID_DEPTH_THRESH = 60000 
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 80.0 
USE_MEDIAN_SCALING = True 

NUM_WORKERS = 8
BATCH_SIZE = 1
# ==============================================

def compute_errors(gt, pred):
    """
    计算指标，增加安全性检查
    """
    # 1. 再次确保没有 0 值 (防止 log 报错)
    pred = torch.clamp(pred, min=1e-4, max=1e4)
    gt = torch.clamp(gt, min=1e-4, max=1e4)

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return torch.tensor([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, gt.numel()], device=gt.device)

def align_scale_shift_torch(pred, target, mask):
    # 仅在双方都有效的地方计算对齐系数
    safe_mask = mask & torch.isfinite(pred) & (pred > 1e-6)
    
    if safe_mask.sum() < 10:
        return pred 
        
    t_median = torch.median(target[safe_mask])
    p_median = torch.median(pred[safe_mask])
    
    if p_median < 1e-6:
        return pred

    ratio = t_median / p_median
    return pred * ratio

class BenchmarkDataset(Dataset):
    def __init__(self, csv_path, pred_root):
        self.df = pd.read_csv(csv_path)
        self.pred_root = Path(pred_root)
        print(f"📊 Dataset Loaded. Total Samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)
    
    def load_gt(self, path):
        if not os.path.exists(path): return None
        depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None: return None
        depth = depth_raw.astype(np.float32)
        depth[depth > INVALID_DEPTH_THRESH] = 0.0
        depth = depth / GLOBAL_SCALE
        return depth

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset_name = row['Dataset']
        renamed_image = row['Renamed_Image']
        gt_path = row['GT_Path']
        
        stem_name = os.path.splitext(renamed_image)[0]
        pred_path = self.pred_root / (stem_name + ".npy")
        
        if not pred_path.exists():
            # 尝试寻找带后缀的
            pred_path = self.pred_root / (renamed_image + ".npy")
            if not pred_path.exists(): return None
        
        try:
            pred = np.load(pred_path) 
            gt = self.load_gt(gt_path)
        except:
            return None

        if gt is None: return None
            
        return torch.from_numpy(pred).float(), torch.from_numpy(gt).float(), {'dataset': dataset_name, 'name': renamed_image}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    preds, gts, metas = zip(*batch)
    return preds, gts, metas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = BenchmarkDataset(args.csv_path, args.pred_root)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    stats = {}
    print(f"🚀 [Step 3] 开始鲁棒评估 (Filter NaNs/Infs)...")
    
    # 调试计数器
    nan_image_count = 0
    
    for batch in tqdm(loader, desc="Evaluating"):
        if batch is None: continue
        preds, gts, metas = batch
        
        for i in range(len(preds)):
            pred = preds[i].to(device)
            gt = gts[i].to(device)
            d_name = metas[i]['dataset']
            img_name = metas[i]['name']
            
            # 1. 检查预测值是否包含 NaN (这是导致 INF 的元凶)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                nan_image_count += 1
                if nan_image_count <= 5: # 只打印前5个报错的，防止刷屏
                    print(f"⚠️ [Warning] Found NaN/Inf in prediction: {img_name} (Dataset: {d_name})")
            
            # 2. 尺寸对齐
            if pred.shape != gt.shape:
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()
            
            # 3. 生成 Mask (关键修改：增加 isfinite 检查)
            # 过滤掉 GT 无效值 AND 预测值无效值 (NaN/Inf)
            mask = (gt > MIN_EVAL_DEPTH) & (gt < MAX_EVAL_DEPTH) & torch.isfinite(pred)
            # --- 🔍 [Debug] 插入诊断代码 ---
            if i == 0 and d_name not in stats: # 每个数据集只打印第一张图
                print(f"\n[Diagnostic] Dataset: {d_name}")
                print(f"  -> GT Path:   {metas[i].get('gt_path', 'N/A')}")
                print(f"  -> GT Range:  Min={gt[mask].min():.4f}, Max={gt[mask].max():.4f}, Mean={gt[mask].mean():.4f}")
                print(f"  -> Pred Range: Min={pred[mask].min():.4f}, Max={pred[mask].max():.4f}, Mean={pred[mask].mean():.4f}")
                
                # 检查单位倍率
                scale_ratio = gt[mask].mean() / (pred[mask].mean() + 1e-6)
                print(f"  -> GT/Pred Ratio: {scale_ratio:.2f}")
                if scale_ratio > 100:
                    print("  ⚠️ 警告: GT 数值远大于预测值，可能是 GT 没除以 1000 (单位是mm)?")
                elif scale_ratio < 0.01:
                    print("  ⚠️ 警告: GT 数值远小于预测值，可能是 GT 被错误地除以了 1000 (单位本来就是m)?")
            if mask.sum() < 10: continue
                
            # 4. 对齐
            if USE_MEDIAN_SCALING:
                pred = align_scale_shift_torch(pred, gt, mask)
            
            # 5. 再次检查对齐后是否出现 NaN (如果 Median 是 NaN)
            if not torch.isfinite(pred).all():
                # 如果对齐后炸了，只取 mask 部分计算，且强行置换 NaN
                pred[~torch.isfinite(pred)] = 0.0 # 这种像素会被下面的 mask 再次过滤吗？
                # 最好的办法是重新 mask
                mask = mask & torch.isfinite(pred)
                if mask.sum() < 10: continue

            # 6. 计算
            pred_valid = pred[mask]
            gt_valid = gt[mask]
            
            metrics = compute_errors(gt_valid, pred_valid)
            
            # 7. 最后一层保险：如果单张图算出 Inf，不要加进去，打印出来
            if torch.isinf(metrics).any() or torch.isnan(metrics).any():
                print(f"❌ [Error] Metric is Inf/NaN for {img_name} | Max GT: {gt_valid.max()}, Min GT: {gt_valid.min()}")
                continue

            # 累加
            if d_name not in stats:
                stats[d_name] = {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse_sum_sq': 0.0, 'rmse_log_sum_sq': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'count': 0}
            
            pixel_count = metrics[7].item()
            stats[d_name]['abs_rel'] += metrics[0].item() * pixel_count
            stats[d_name]['sq_rel']  += metrics[1].item() * pixel_count
            stats[d_name]['rmse_sum_sq'] += (metrics[2].item() ** 2) * pixel_count
            stats[d_name]['rmse_log_sum_sq'] += (metrics[3].item() ** 2) * pixel_count
            stats[d_name]['a1'] += metrics[4].item() * pixel_count
            stats[d_name]['a2'] += metrics[5].item() * pixel_count
            stats[d_name]['a3'] += metrics[6].item() * pixel_count
            stats[d_name]['count'] += pixel_count

    # ================= 📊 生成报告 =================
    print("\n" + "="*110)
    header = f"{'Dataset':<15} | {'Count':<8} | {'AbsRel':<8} | {'SqRel':<8} | {'RMSE':<8} | {'RMSElog':<8} | {'a1':<8} | {'a2':<8} | {'a3':<8}"
    print(header)
    print("-" * 110)
    
    report_lines = [header, "-" * 110]
    overall_stats = {k: 0.0 for k in ['abs_rel', 'sq_rel', 'rmse_sum_sq', 'rmse_log_sum_sq', 'a1', 'a2', 'a3', 'count']}

    for d_name in sorted(stats.keys()):
        data = stats[d_name]
        count = data['count']
        for k in overall_stats: overall_stats[k] += data[k]
        
        if count > 0:
            avg = {
                'abs_rel': data['abs_rel'] / count,
                'sq_rel': data['sq_rel'] / count,
                'rmse': np.sqrt(data['rmse_sum_sq'] / count),
                'rmse_log': np.sqrt(data['rmse_log_sum_sq'] / count),
                'a1': data['a1'] / count,
                'a2': data['a2'] / count,
                'a3': data['a3'] / count
            }
            img_count = len(dataset.df[dataset.df['Dataset'] == d_name])
            line = f"{d_name:<15} | {img_count:<8} | {avg['abs_rel']:.4f}   | {avg['sq_rel']:.4f}   | {avg['rmse']:.4f}   | {avg['rmse_log']:.4f}   | {avg['a1']:.3f}    | {avg['a2']:.3f}    | {avg['a3']:.3f}"
            print(line)
            report_lines.append(line)

    print("-" * 110)
    if overall_stats['count'] > 0:
        c = overall_stats['count']
        o_avg = {
            'abs_rel': overall_stats['abs_rel'] / c,
            'sq_rel': overall_stats['sq_rel'] / c,
            'rmse': np.sqrt(overall_stats['rmse_sum_sq'] / c),
            'rmse_log': np.sqrt(overall_stats['rmse_log_sum_sq'] / c),
            'a1': overall_stats['a1'] / c,
            'a2': overall_stats['a2'] / c,
            'a3': overall_stats['a3'] / c
        }
        line = f"{'OVERALL':<15} | {len(dataset):<8} | {o_avg['abs_rel']:.4f}   | {o_avg['sq_rel']:.4f}   | {o_avg['rmse']:.4f}   | {o_avg['rmse_log']:.4f}   | {o_avg['a1']:.3f}    | {o_avg['a2']:.3f}    | {o_avg['a3']:.3f}"
        print(line)
        report_lines.append("-" * 110)
        report_lines.append(line)

    with open(args.output_dir + "/Final_Report.txt", 'w') as f: f.write('\n'.join(report_lines))
    
    if nan_image_count > 0:
        print(f"\n⚠️ 警告: 共发现 {nan_image_count} 张图片包含 NaN/Inf 预测值，已自动跳过这些像素。")
        print(f"💡 建议: 如果 NaN 数量很多，请在 Step 1 推理脚本中把 fp16=True 改为 fp16=False 重新推理。")

if __name__ == "__main__":
    main()