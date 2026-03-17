# import numpy as np
# import os
# from glob import glob
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from datetime import datetime
# import warnings

# # 忽略一些除以0的警告 (常见于深度图处理)
# warnings.filterwarnings("ignore")

# # =================================================================================
# # 配置区域
# # =================================================================================

# # 核心开关：是否是 Depth Anything V2 这种相对视差模型？
# # 如果是 True，脚本会先执行 pred = 1.0 / (pred + 1e-6) 再对齐
# IS_RELATIVE_DISPARITY_MODEL = True 

# BATCH_SIZE = 16  # 建议根据显存调整，Scale&Shift需要一些额外显存做矩阵运算
# NUM_WORKERS = 8
# MIN_EVAL_DEPTH = 1e-3
# MAX_EVAL_DEPTH = 500  # 评估的最大距离 (训练可能是1000，评估通常关注有效区)

# # 输出结果路径
# OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-ScaleShift-b.txt"

# # 数据集路径配置 (请在此处填入你的 DAV2 预测路径和 GT 路径)
# EVALUATION_PATHS = [
#     # 示例: 你可以把之前的路径复制过来，或者填入新的
#     {
#         'name': 'dj3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/dj3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj3/npy'
#     },
#         {
#         'name': 'hsd1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/hsd1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/hsd1/npy'
#     },
#         {
#         'name': 'xg5',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/xg5',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg5/npy'
#     },


#             {
#         'name': 'lower',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/lower',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/lower-npy'
#     },

#             {
#         'name': 'sziit',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/sziit',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/sziit-npy'
#     },


#             {
#         'name': 'tonw1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/town1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town1-npy'
#     },
#             {
#         'name': 'town2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/town2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town2-npy'
#     },
#             {
#         'name': 'town3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/town3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town3-npy'
#     },


#             {
#         'name': 'yingrenshi1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/yingrenshi1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi1-npy'
#     },
#             {
#         'name': 'yingrenshi2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b/yingrenshi2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi2-npy'
#     },


# ]

# # =================================================================================
# # DataLoader 部分 (保持不变，只负责 I/O)
# # =================================================================================

# class PairedNpyDataset(Dataset):
#     def __init__(self, pred_dir, gt_dir):
#         # 递归查找或单层查找，取决于你的目录结构。这里用单层查找以保持一致性
#         # 如果文件名完全一致
#         all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
#         self.pred_files = []
#         self.gt_files = []
        
#         for pred_path in all_pred_files:
#             basename = os.path.basename(pred_path)
#             gt_path = os.path.join(gt_dir, basename)
            
#             # 如果文件名不匹配(比如多了前缀)，请在这里修改 gt_path 的生成逻辑
#             if os.path.exists(gt_path):
#                 self.pred_files.append(pred_path)
#                 self.gt_files.append(gt_path)
        
#         if len(self.pred_files) == 0:
#             # 尝试递归查找作为备选方案
#             all_pred_files = sorted(glob(os.path.join(pred_dir, "**/*.npy"), recursive=True))
#             for pred_path in all_pred_files:
#                 basename = os.path.basename(pred_path)
#                 # 假设 GT 目录结构比较扁平，或者你需要根据 pred_path 解析 gt_path
#                 # 这里为了简单，假设文件名唯一
#                 possible_gts = glob(os.path.join(gt_dir, "**", basename), recursive=True)
#                 if len(possible_gts) > 0:
#                      self.pred_files.append(pred_path)
#                      self.gt_files.append(possible_gts[0])

#         if len(self.pred_files) == 0:
#              print(f"警告: 在 {pred_dir} 未找到匹配文件。请检查文件名是否一致。")

#     def __len__(self):
#         return len(self.pred_files)

#     def __getitem__(self, idx):
#         try:
#             pred_np = np.load(self.pred_files[idx]).astype(np.float32)
#             gt_np = np.load(self.gt_files[idx]).astype(np.float32)

#             if pred_np.ndim == 3: pred_np = np.squeeze(pred_np)
#             if gt_np.ndim == 3: gt_np = np.squeeze(gt_np)
            
#             return torch.from_numpy(pred_np), torch.from_numpy(gt_np)
#         except Exception as e:
#             print(f"Error loading {self.pred_files[idx]}: {e}")
#             # 返回全0张量避免崩溃，后续会被 mask 掉
#             return torch.zeros((1,1)), torch.zeros((1,1))

# # =================================================================================
# # 核心逻辑: 向量化误差计算 (保持不变)
# # =================================================================================

# def compute_errors_torch(gt, pred, mask):
#     """
#     输入:
#     - gt: (B, H, W) 真实深度
#     - pred: (B, H, W) 已经对齐好的预测深度
#     - mask: (B, H, W) 有效像素掩码
#     """
#     # 1. 统计有效像素
#     valid_pixel_count = mask.sum(dim=[1, 2]).to(torch.float32)

#     # 2. 预处理 (防止 log(0) 或 除0)
#     # 计算误差时，我们只关心 mask 内的区域，但 clamp 防止 nan 扩散
#     gt_clamped = torch.clamp(gt, min=1e-3)
#     pred_clamped = torch.clamp(pred, min=1e-3)

#     # 3. 阈值准确率 (a1, a2, a3)
#     thresh = torch.maximum((gt_clamped / pred_clamped), (pred_clamped / gt_clamped))
    
#     a1_count = ((thresh < 1.25) & mask).sum(dim=[1, 2]).to(torch.float32)
#     a2_count = ((thresh < 1.25 ** 2) & mask).sum(dim=[1, 2]).to(torch.float32)
#     a3_count = ((thresh < 1.25 ** 3) & mask).sum(dim=[1, 2]).to(torch.float32)

#     # 4. RMSE
#     rmse_map = (gt - pred) ** 2
#     rmse_map[~mask] = 0
#     rmse_sum_sq = rmse_map.sum(dim=[1, 2])

#     # 5. RMSE log
#     rmse_log_map = (torch.log(gt_clamped) - torch.log(pred_clamped)) ** 2
#     rmse_log_map[~mask] = 0
#     rmse_log_sum_sq = rmse_log_map.sum(dim=[1, 2])

#     # 6. AbsRel
#     abs_rel_map = torch.abs(gt - pred) / gt_clamped
#     abs_rel_map[~mask] = 0
#     abs_rel_sum = abs_rel_map.sum(dim=[1, 2])

#     # 7. SqRel
#     sq_rel_map = ((gt - pred) ** 2) / gt_clamped
#     sq_rel_map[~mask] = 0
#     sq_rel_sum = sq_rel_map.sum(dim=[1, 2])

#     # 组装
#     results = torch.stack([
#         abs_rel_sum, sq_rel_sum, rmse_sum_sq, rmse_log_sum_sq, 
#         a1_count, a2_count, a3_count, valid_pixel_count
#     ], dim=1)
    
#     return results

# def compute_metrics_from_sums(errors_sum_array):
#     total_sums = errors_sum_array
#     total_valid_pixels = total_sums[7]
#     if total_valid_pixels == 0: return np.zeros(7)
    
#     final_metrics = np.zeros(7)
#     final_metrics[0] = total_sums[0] / total_valid_pixels # abs_rel
#     final_metrics[1] = total_sums[1] / total_valid_pixels # sq_rel
#     final_metrics[2] = np.sqrt(total_sums[2] / total_valid_pixels) # rmse
#     final_metrics[3] = np.sqrt(total_sums[3] / total_valid_pixels) # rmse_log
#     final_metrics[4] = total_sums[4] / total_valid_pixels # a1
#     final_metrics[5] = total_sums[5] / total_valid_pixels # a2
#     final_metrics[6] = total_sums[6] / total_valid_pixels # a3
#     return final_metrics

# # =================================================================================
# # 核心逻辑: 批量对齐与评估 (Scale & Shift 版本)
# # =================================================================================

# def batch_least_squares_alignment(pred_inv, gt, mask):
#     """
#     针对 Batch 数据的 Scale & Shift 对齐
#     solve: s * pred_inv + t = gt
    
#     由于每个样本的 mask 不同，难以完全向量化为一个矩阵操作，
#     这里使用循环处理 Batch 维度 (Batch通常很小，如16/32，速度影响可忽略)
#     """
#     B = pred_inv.shape[0]
#     aligned_pred = torch.zeros_like(pred_inv)
    
#     for i in range(B):
#         # 取出单张图的有效像素
#         valid_mask = mask[i]
#         if valid_mask.sum() < 10: # 像素太少，跳过对齐
#             aligned_pred[i] = pred_inv[i] 
#             continue
            
#         y = gt[i][valid_mask]       # (N,)
#         x = pred_inv[i][valid_mask] # (N,)
        
#         # 构造最小二乘矩阵 [x, 1]
#         # stack: (N, 2)
#         ones = torch.ones_like(x)
#         A = torch.stack([x, ones], dim=1)
        
#         # 求解 (2,) -> [s, t]
#         # lstsq 返回: solution, residuals, rank, singular_values
#         solution = torch.linalg.lstsq(A, y).solution
#         s, t = solution[0], solution[1]
        
#         # 应用全局
#         aligned_pred[i] = pred_inv[i] * s + t
        
#     return aligned_pred

# def evaluate_single_scene(pred_dir, gt_dir, scene_name, num_workers, batch_size=16, min_depth=1e-3, max_depth=1000):
#     device = torch.device("cuda")
    
#     dataset = PairedNpyDataset(pred_dir, gt_dir)
#     if len(dataset) == 0: return None
    
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     all_errors_scene = []
    
#     for pred_batch, gt_batch in tqdm(dataloader, desc=f"评估: {scene_name}", leave=False):
#         pred_batch, gt_batch = pred_batch.to(device), gt_batch.to(device)
        
#         # 1. 尺寸对齐 (Interpolate)
#         target_shape = gt_batch.shape[2:]
#         if pred_batch.shape[2:] != target_shape:
#             pred_batch = pred_batch.unsqueeze(1)
#             pred_batch = F.interpolate(pred_batch, size=target_shape, mode='bilinear', align_corners=False)
#             pred_batch = pred_batch.squeeze(1)

#         # 2. 生成掩码 (GT Valid Mask)
#         mask_batch = (gt_batch > min_depth) & (gt_batch < max_depth) & torch.isfinite(gt_batch)

#         # 3. 核心转换与对齐逻辑
#         if IS_RELATIVE_DISPARITY_MODEL:
#             # A. 取倒数: Disparity -> Depth Space
#             # 注意: 这里加 1e-6 防止除以0
#             pred_process = 1.0 / (pred_batch + 1e-6)
            
#             # B. Scale & Shift 对齐
#             # 这一步计算 s, t 并应用
#             pred_aligned = batch_least_squares_alignment(pred_process, gt_batch, mask_batch)
            
#             # C. 防止对齐后出现负数 (因为 Shift 可能是负的)
#             pred_aligned.clamp_(min=1e-3)
#         else:
#             # 如果是绝对模型但想做 Scale&Shift，直接传
#             pred_aligned = batch_least_squares_alignment(pred_batch, gt_batch, mask_batch)
#             pred_aligned.clamp_(min=1e-3)

#         # 4. 计算误差 (使用对齐后的 pred_aligned)
#         errors_batch = compute_errors_torch(gt_batch, pred_aligned, mask_batch)
#         all_errors_scene.append(errors_batch.cpu().numpy())
            
#     return np.concatenate(all_errors_scene, axis=0) if all_errors_scene else None

# def format_results(mean_errors, name):
#     header = (" {:<65} |" + " {:>8} |" * 7).format(name, "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
#     line = ("-" * 67 + "|") + ("-" * 10 + "|") * 7
#     values = (" {:<65} |" + "&{: 8.3f} |" * 7).format(name, *mean_errors.tolist())
#     return f"{header}\n{line}\n{values}\n"

# # =================================================================================
# # 主程序
# # =================================================================================

# if __name__ == '__main__':
#     if not torch.cuda.is_available():
#         print("错误: 未检测到CUDA GPU。程序终止。")
#         exit()
        
#     results_per_scene = {}
#     all_errors_combined = []
    
#     print("="*100)
#     print(f"深度评估工具 - Scale & Shift 版 (适配 Depth Anything V2)")
#     print(f"模式: {'Relative Disparity (1/x -> Alignment)' if IS_RELATIVE_DISPARITY_MODEL else 'Linear Alignment Only'}")
#     print(f"范围: {MIN_EVAL_DEPTH}m - {MAX_EVAL_DEPTH}m")
#     print("="*100)

#     for path_info in tqdm(EVALUATION_PATHS, desc="总进度"):
#         scene_name = path_info.get('name', 'unknown')
#         pred_dir = path_info['pred_dir']
#         gt_dir = path_info['gt_dir']

#         if not os.path.exists(pred_dir):
#             print(f"Skipping {scene_name}, pred_dir missing: {pred_dir}")
#             continue

#         scene_errors = evaluate_single_scene(
#             pred_dir, gt_dir, scene_name,
#             num_workers=NUM_WORKERS,
#             batch_size=BATCH_SIZE,
#             min_depth=MIN_EVAL_DEPTH,
#             max_depth=MAX_EVAL_DEPTH
#         )
        
#         if scene_errors is not None:
#             results_per_scene[scene_name] = scene_errors
#             all_errors_combined.append(scene_errors)

#     # 生成报告
#     report_lines = []
#     report_lines.append("=" * 145)
#     report_lines.append(f"评估报告 - {datetime.now()}")
#     report_lines.append(f"Method: Least Squares Alignment (Scale & Shift)")
#     report_lines.append(f"Input Mode: {'Disparity (Inverted)' if IS_RELATIVE_DISPARITY_MODEL else 'Depth'}")
#     report_lines.append("=" * 145)
    
#     sorted_names = sorted(results_per_scene.keys())
#     for name in sorted_names:
#         metrics = compute_metrics_from_sums(results_per_scene[name].sum(axis=0))
#         report_lines.append(format_results(metrics, name))
        
#     if all_errors_combined:
#         all_sum = np.concatenate(all_errors_combined, axis=0).sum(axis=0)
#         overall_metrics = compute_metrics_from_sums(all_sum)
#         report_lines.append("=" * 145)
#         report_lines.append(format_results(overall_metrics, "===> Overall Average"))
#         report_lines.append("=" * 145)
        
#     final_report = "\n".join(report_lines)
#     print("\n" + final_report)
    
#     os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
#     with open(OUTPUT_FILENAME, 'w') as f:
#         f.write(final_report)
#     print(f"\nSaved to: {OUTPUT_FILENAME}")


import numpy as np
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import warnings

# 忽略一些除以0的警告 (常见于深度图处理)
warnings.filterwarnings("ignore")

# =================================================================================
# 1. 路径列表定义区域 (在这里粘贴你的路径组)
# =================================================================================

# 示例：Depth Anything V2 Base 的输出路径
PATHS_DAV2_BASE = [
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-b/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-b/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
    },
    # ... 其他场景 ...
]

PATHS_DAV2_LARGE = [
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-l/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-l/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
    },
    # ... 其他场景 ...
]


PATHS_DAV2_SMALL = [
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-s/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-s/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
    },
    # ... 其他场景 ...
]

# =================================================================================
# 2. 任务配置区域 (TASKS)
# =================================================================================

TASKS = [
    # --- 任务 1: Depth Anything V2 Base (相对视差模式) ---
    # {
    #     "task_name": "DAV2-Base-ScaleShift",
    #     "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer2-ScaleShift-b.txt",
        
    #     # 关键开关: True 表示这是视差模型(如DepthAnything)，需要先取倒数(1/x)再对齐
    #     #          False 表示这是深度模型(如Metric3D)，直接进行 Scale & Shift 对齐
    #     "use_relative_disparity_mode": True, 
        
    #     "paths": PATHS_DAV2_BASE  # 引用上面的列表
    # },
      {
        "task_name": "DAV2-LARGE-ScaleShift",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer2-ScaleShift-l.txt",
        
        # 关键开关: True 表示这是视差模型(如DepthAnything)，需要先取倒数(1/x)再对齐
        #          False 表示这是深度模型(如Metric3D)，直接进行 Scale & Shift 对齐
        "use_relative_disparity_mode": True, 
        
        "paths": PATHS_DAV2_LARGE  # 引用上面的列表
    },
      {
        "task_name": "DAV2-SMALL-ScaleShift",
        "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-ScaleShift-s.txt",
        
        # 关键开关: True 表示这是视差模型(如DepthAnything)，需要先取倒数(1/x)再对齐
        #          False 表示这是深度模型(如Metric3D)，直接进行 Scale & Shift 对齐
        "use_relative_disparity_mode": True, 
        
        "paths": PATHS_DAV2_SMALL  # 引用上面的列表
    },

    # --- 任务 2: 示例 (如果有 Large 模型) ---
    # {
    #     "task_name": "DAV2-Large-ScaleShift",
    #     "output_file": "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-ScaleShift-l.txt",
    #     "use_relative_disparity_mode": True, 
    #     "paths": PATHS_DAV2_LARGE
    # },
]

# =================================================================================
# 核心设置 (通常无需修改)
# =================================================================================
BATCH_SIZE = 16  # Scale&Shift 矩阵运算比较吃显存，保持 16 比较安全
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 500

# =================================================================================
# 数据加载器
# =================================================================================

class PairedNpyDataset(Dataset):
    def __init__(self, pred_dir, gt_dir):
        # 优先单层查找
        all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        self.pred_files = []
        self.gt_files = []
        
        for pred_path in all_pred_files:
            basename = os.path.basename(pred_path)
            gt_path = os.path.join(gt_dir, basename)
            if os.path.exists(gt_path):
                self.pred_files.append(pred_path)
                self.gt_files.append(gt_path)
        
        # 备选：递归查找 (如果第一步没找到)
        if len(self.pred_files) == 0:
            all_pred_files = sorted(glob(os.path.join(pred_dir, "**/*.npy"), recursive=True))
            for pred_path in all_pred_files:
                basename = os.path.basename(pred_path)
                possible_gts = glob(os.path.join(gt_dir, "**", basename), recursive=True)
                if len(possible_gts) > 0:
                     self.pred_files.append(pred_path)
                     self.gt_files.append(possible_gts[0])

        if len(self.pred_files) == 0:
             print(f"警告: 在 {pred_dir} 未找到匹配文件。")

    def __len__(self): return len(self.pred_files)

    def __getitem__(self, idx):
        try:
            pred_np = np.load(self.pred_files[idx]).astype(np.float32)
            gt_np = np.load(self.gt_files[idx]).astype(np.float32)
            if pred_np.ndim == 3: pred_np = np.squeeze(pred_np)
            if gt_np.ndim == 3: gt_np = np.squeeze(gt_np)
            return torch.from_numpy(pred_np), torch.from_numpy(gt_np)
        except Exception as e:
            print(f"Error loading {self.pred_files[idx]}: {e}")
            return torch.zeros((1,1)), torch.zeros((1,1))

# =================================================================================
# 误差计算函数
# =================================================================================

def compute_errors_torch(gt, pred, mask):
    valid_pixel_count = mask.sum(dim=[1, 2]).to(torch.float32)
    
    # 预处理
    gt_clamped = torch.clamp(gt, min=1e-3)
    pred_clamped = torch.clamp(pred, min=1e-3)

    # 阈值准确率
    thresh = torch.maximum((gt_clamped / pred_clamped), (pred_clamped / gt_clamped))
    a1_count = ((thresh < 1.25) & mask).sum(dim=[1, 2]).float()
    a2_count = ((thresh < 1.25 ** 2) & mask).sum(dim=[1, 2]).float()
    a3_count = ((thresh < 1.25 ** 3) & mask).sum(dim=[1, 2]).float()

    # RMSE
    rmse_map = (gt - pred) ** 2; rmse_map[~mask] = 0
    rmse_sum_sq = rmse_map.sum(dim=[1, 2])

    # RMSE log
    rmse_log_map = (torch.log(gt_clamped) - torch.log(pred_clamped)) ** 2; rmse_log_map[~mask] = 0
    rmse_log_sum_sq = rmse_log_map.sum(dim=[1, 2])

    # AbsRel
    abs_rel_map = torch.abs(gt - pred) / gt_clamped; abs_rel_map[~mask] = 0
    abs_rel_sum = abs_rel_map.sum(dim=[1, 2])

    # SqRel
    sq_rel_map = ((gt - pred) ** 2) / gt_clamped; sq_rel_map[~mask] = 0
    sq_rel_sum = sq_rel_map.sum(dim=[1, 2])

    return torch.stack([
        abs_rel_sum, sq_rel_sum, rmse_sum_sq, rmse_log_sum_sq, 
        a1_count, a2_count, a3_count, valid_pixel_count
    ], dim=1)

def compute_metrics_from_sums(sums):
    total = sums[7]
    if total == 0: return np.zeros(7)
    return np.array([
        sums[0]/total, sums[1]/total, np.sqrt(sums[2]/total), np.sqrt(sums[3]/total),
        sums[4]/total, sums[5]/total, sums[6]/total
    ])

# =================================================================================
# 批量 Scale & Shift 对齐逻辑
# =================================================================================

def batch_least_squares_alignment(pred_inv, gt, mask):
    """
    solve: s * pred_inv + t = gt
    """
    B = pred_inv.shape[0]
    aligned_pred = torch.zeros_like(pred_inv)
    
    for i in range(B):
        valid_mask = mask[i]
        if valid_mask.sum() < 10: 
            aligned_pred[i] = pred_inv[i] 
            continue
            
        y = gt[i][valid_mask]       # (N,)
        x = pred_inv[i][valid_mask] # (N,)
        
        # 构造矩阵 [x, 1]
        ones = torch.ones_like(x)
        A = torch.stack([x, ones], dim=1)
        
        # 最小二乘求解
        solution = torch.linalg.lstsq(A, y).solution
        s, t = solution[0], solution[1]
        
        aligned_pred[i] = pred_inv[i] * s + t
        
    return aligned_pred

def evaluate_single_scene(pred_dir, gt_dir, scene_name, is_relative_mode):
    device = torch.device("cuda")
    
    dataset = PairedNpyDataset(pred_dir, gt_dir)
    if len(dataset) == 0: return None
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    all_errors_scene = []
    
    for pred_batch, gt_batch in tqdm(dataloader, desc=f"  -> {scene_name}", leave=False):
        pred_batch, gt_batch = pred_batch.to(device), gt_batch.to(device)
        
        # 1. 尺寸对齐
        target_shape = gt_batch.shape[2:]
        if pred_batch.shape[2:] != target_shape:
            pred_batch = F.interpolate(pred_batch.unsqueeze(1), size=target_shape, mode='bilinear', align_corners=False).squeeze(1)

        # 2. 生成掩码
        mask_batch = (gt_batch > MIN_EVAL_DEPTH) & (gt_batch < MAX_EVAL_DEPTH) & torch.isfinite(gt_batch)

        # 3. 对齐处理
        if is_relative_mode:
            # 相对视差模式 (DepthAnything): 先取倒数转深度，再 Scale & Shift
            pred_process = 1.0 / (pred_batch + 1e-6)
            pred_aligned = batch_least_squares_alignment(pred_process, gt_batch, mask_batch)
            pred_aligned.clamp_(min=1e-3)
        else:
            # 绝对深度模式 (Metric3D等): 直接 Scale & Shift
            pred_aligned = batch_least_squares_alignment(pred_batch, gt_batch, mask_batch)
            pred_aligned.clamp_(min=1e-3)

        # 4. 计算误差
        errors_batch = compute_errors_torch(gt_batch, pred_aligned, mask_batch)
        all_errors_scene.append(errors_batch.cpu().numpy())
            
    return np.concatenate(all_errors_scene, axis=0) if all_errors_scene else None

def format_line(name, m):
    return "{:<50} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} |".format(
        name[-50:], m[0], m[1], m[2], m[3], m[4], m[5], m[6])

# =================================================================================
# 主循环
# =================================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available(): exit("No CUDA device.")

    print(f"Starting Multi-Task Evaluation (Scale & Shift)")
    print(f"Total Tasks: {len(TASKS)}")

    for i, task in enumerate(TASKS):
        t_name = task['task_name']
        t_out = task['output_file']
        t_paths = task['paths']
        # 获取当前任务的模式开关 (默认为 True 以防忘记写)
        t_relative = task.get('use_relative_disparity_mode', True) 

        print(f"\n[{i+1}/{len(TASKS)}] Processing Task: {t_name}")
        print(f"  Mode: {'Relative Disparity (1/x + Alignment)' if t_relative else 'Depth Alignment Only'}")
        print(f"  Output: {t_out}")
        print(f"  Scenes: {len(t_paths)}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(t_out), exist_ok=True)
        
        results_map = {}
        all_errs_list = []
        
        for path_item in t_paths:
            s_name = path_item['name']
            p_dir = path_item['pred_dir']
            g_dir = path_item['gt_dir']
            
            if not os.path.exists(p_dir):
                print(f"  [Error] Path missing: {p_dir}")
                continue

            # 传入 t_relative 参数
            errs = evaluate_single_scene(p_dir, g_dir, s_name, is_relative_mode=t_relative)
            
            if errs is not None:
                results_map[s_name] = errs
                all_errs_list.append(errs)

        # 生成报告
        lines = []
        header = "{:<50} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |".format("Scene", "AbsRel", "SqRel", "RMSE", "RMSElog", "a1", "a2", "a3")
        sep = "-" * len(header)
        
        lines += [
            "="*100,
            f"Task: {t_name}", 
            f"Date: {datetime.now()}", 
            f"Alignment: Scale & Shift (Least Squares)",
            f"Input Mode: {'Disparity (1/x)' if t_relative else 'Depth'}", 
            "="*100, 
            header, 
            sep
        ]
        
        for s_name in sorted(results_map.keys()):
            s_mean = compute_metrics_from_sums(results_map[s_name].sum(0))
            lines.append(format_line(s_name, s_mean))
            
        if all_errs_list:
            total_mean = compute_metrics_from_sums(np.concatenate(all_errs_list, 0).sum(0))
            lines += [sep, format_line(">>> OVERALL AVERAGE", total_mean), "="*100]
            
        with open(t_out, 'w') as f: f.write("\n".join(lines))
        print(f"  -> Report saved to {t_out}")
        
        # 显存清理
        del results_map, all_errs_list
        torch.cuda.empty_cache()

    print("\nAll tasks completed.")