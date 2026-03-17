


import numpy as np
import cv2
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# =================================================================================
# 配置区域 (与您的一致)
# =================================================================================
USE_MEDIAN_SCALING = False
BATCH_SIZE = 32
# <<< 建议: 优化后, CPU瓶颈可能转移到I/O, 您可以尝试将其设为 12 或 16 (取决于您的CPU核心数) >>>
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 1000




#DAV2
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-b"
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-s"
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Infer-l"

#M3Dv2
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/metric3D/D1/Infer"


#Moge2
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/Infer"
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-Moge-nonormal"
OUTPUT_FILENAME = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-Moge-nonormal"

EVALUATION_PATHS = [


  
# {
#     'name': 'hav',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-out/hav',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_hav'
# },
{
    'name': 'lfls',
    'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-4k-out/lfls',
    'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_lfls'
},
# {
#     'name': 'lfls2',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-out/lfls2',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_lfls2'
# },
{
    'name': 'lower',
    'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-4k-out/lower',
    'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_lower'
},
# {
#     'name': 'SMBU',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-out/SMBU',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_SMBU'
# },
# {
#     'name': 'sziit',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-4k-out/sziit',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_sziit'
# },
# {
#     'name': 'sztu',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-out/sztu',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_sztu'
# },
# {
#     'name': 'upper',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GAU-out/upper',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_upper'
# }




# {
#     'name': 'dj',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/dj',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj/npy'
# },
# {
#     'name': 'dj2',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/dj2',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj2/npy'
# },
# {
#     'name': 'dj3',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/dj3',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj3/npy'
# },
# {
#     'name': 'dj4',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/dj4',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj4/npy'
# },
# {
#     'name': 'hsd1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/hsd1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/hsd1/npy'
# },
# {
#     'name': 'lm',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/lm',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/lm/npy'
# },
# {
#     'name': 'xg1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/xg1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg1/npy'
# },
# {
#     'name': 'xg2',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/xg2',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg2/npy'
# },
# {
#     'name': 'xg3',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/xg3',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg3/npy'
# },
# {
#     'name': 'xg4',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/xg4',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg4/npy'
# },
# {
#     'name': 'xg5',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Moge2/GES11-out/xg5',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg5/npy'
# }
]

    

# =================================================================================
# 优化部分 1: DataLoader 仅负责 I/O
# =================================================================================

class PairedNpyDataset(Dataset):
    def __init__(self, pred_dir, gt_dir):
        all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        self.pred_files = []
        self.gt_files = []
        for pred_path in all_pred_files:
            basename = os.path.splitext(os.path.basename(pred_path))[0]
            gt_filename = f"{basename}.npy"
            gt_path = os.path.join(gt_dir, gt_filename)
            if os.path.exists(gt_path):
                self.pred_files.append(pred_path)
                self.gt_files.append(gt_path)
                
        if len(self.pred_files) == 0:
            raise ValueError(f"在 '{pred_dir}' 和 '{gt_dir}' 之间未找到任何匹配的文件对。")
        
        # <<< 优化: 移除在 __init__ 中加载样本和确定形状的逻辑 >>>
        # CPU worker 进程不应该在初始化时加载数据
        
        print(f"       -> 找到 {len(self.pred_files)} 个有效的 PRED/GT 文件对。")
        print(f"       -> 目标插值形状将在运行时 (GPU上) 动态确定。")

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        # __getitem__ 只负责最快的 I/O 和 Numpy 转换
        pred_np = np.load(self.pred_files[idx]).astype(np.float32)
        gt_np = np.load(self.gt_files[idx]).astype(np.float32)

        if pred_np.ndim == 3:
            pred_np = np.squeeze(pred_np)
        if gt_np.ndim == 3:
            gt_np = np.squeeze(gt_np)
            
        pred_tensor = torch.from_numpy(pred_np)
        gt_tensor = torch.from_numpy(gt_np)
        
        # <<< 优化: 移除 F.interpolate (插值) >>>
        # 这个操作非常耗时，将其移出 DataLoader，放到 GPU 上执行
            
        return pred_tensor, gt_tensor

# =================================================================================
# 优化部分 2: 核心代码向量化 (Vectorization)
# =================================================================================

def compute_errors_torch(gt, pred, mask):
    """
    MODIFIED (Vectorized):
    计算误差 *总和* 和 *像素计数*，而不是平均值。
    此版本完全向量化，移除了 BATCH 上的 for 循环，速度极快。
    
    输入:
    - gt: (B, H, W)
    - pred: (B, H, W)
    - mask: (B, H, W) (bool)
    
    返回: (B, 8) 张量, 8 列为:
    [abs_rel_sum, sq_rel_sum, rmse_sum_sq, rmse_log_sum_sq, a1_count, a2_count, a3_count, valid_pixel_count]
    """
    B = gt.shape[0]
    
    # 1. 统计有效像素 (B,)
    # .sum() 在 bool 张量上工作，然后转为 float
    valid_pixel_count = mask.sum(dim=[1, 2]).to(torch.float32)

    # 2. 裁剪 (B, H, W)
    # 我们只裁剪用于除法和log的gt/pred
    # 注意：这里我们使用 gt_clamped = torch.clamp(gt, min=1e-3) 
    # 但在计算误差时（如RMSE），我们仍使用原始的 gt 和 pred
    gt_clamped = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_clamped = torch.clamp(pred, min=MIN_EVAL_DEPTH)

    # # 3. 阈值 (Thresh) (B, H, W)
    # # 使用裁剪后的值计算 thresh
    # thresh = torch.maximum((gt_clamped / pred_clamped), (pred_clamped / gt_clamped))
    # # 应用掩码 (将无效像素设为1, 这样它们对 'a' 计数的 < 1.25 比较为 False)
    # thresh[~mask] = 1.25 # 设为 1.25 或更大值，使其在 a1 比较中为 False

    # # 4. a1, a2, a3 (B,)
    # # 直接在 (B, H, W) 张量上计算，然后沿 H, W 维度求和
    # a1_count = (thresh < 1.25).sum(dim=[1, 2]).to(torch.float32)
    # a2_count = (thresh < 1.25 ** 2).sum(dim=[1, 2]).to(torch.float32)
    # a3_count = (thresh < 1.25 ** 3).sum(dim=[1, 2]).to(torch.float32)


# 3. 阈值 (Thresh) (B, H, W)
    # 使用裁剪后的值计算 thresh
    thresh = torch.maximum((gt_clamped / pred_clamped), (pred_clamped / gt_clamped))
    
    # <<< 修复: 删除了 'thresh[~mask] = 1.25' 这一行 >>>
    
    # 4. a1, a2, a3 (B,)
    # <<< 修复: 我们只统计 掩码(mask)为True 且 阈值(thresh)通过 的像素 >>>
    # 使用 '& mask' 确保只计算有效像素
    a1_count = ((thresh < 1.25) & mask).sum(dim=[1, 2]).to(torch.float32)
    a2_count = ((thresh < 1.25 ** 2) & mask).sum(dim=[1, 2]).to(torch.float32)
    a3_count = ((thresh < 1.25 ** 3) & mask).sum(dim=[1, 2]).to(torch.float32)
    # --- 计算误差图 (Error Maps) ---
    # 我们先计算 (B, H, W) 的误差图，然后将无效区域 [~mask] 设为 0，最后再求和。
    
    # 5. RMSE (B, H, W) -> (B,)
    rmse_map = (gt - pred) ** 2
    rmse_map[~mask] = 0 # 将无效像素的误差设为0
    rmse_sum_sq = rmse_map.sum(dim=[1, 2])

    # 6. RMSE_log (B, H, W) -> (B,)
    # 使用裁剪后的值进行log
    rmse_log_map = (torch.log(gt_clamped) - torch.log(pred_clamped)) ** 2
    rmse_log_map[~mask] = 0 # 将无效像素的误差设为0
    rmse_log_sum_sq = rmse_log_map.sum(dim=[1, 2])

    # 7. AbsRel (B, H, W) -> (B,)
    # 使用裁剪后的gt进行除法，防止除以0
    abs_rel_map = torch.abs(gt - pred) / gt_clamped
    abs_rel_map[~mask] = 0
    abs_rel_sum = abs_rel_map.sum(dim=[1, 2])

    # 8. SqRel (B, H, W) -> (B,)
    # 使用裁剪后的gt进行除法
    sq_rel_map = ((gt - pred) ** 2) / gt_clamped
    sq_rel_map[~mask] = 0
    sq_rel_sum = sq_rel_map.sum(dim=[1, 2])

    # 9. 组装结果 (B, 8)
    # 使用 torch.stack 将 (B,) 的张量们堆叠成 (B, 8)
    results = torch.stack([
        abs_rel_sum, 
        sq_rel_sum, 
        rmse_sum_sq, 
        rmse_log_sum_sq, 
        a1_count, 
        a2_count, 
        a3_count, 
        valid_pixel_count
    ], dim=1) # dim=1 表示在第二维上堆叠
    
    return results

# (compute_metrics_from_sums 函数无需修改, 它已经很快了)
def compute_metrics_from_sums(errors_sum_array):
    """
    输入一个 (8,) 数组，包含所有误差总和与总像素数。
    返回一个 (7,) 数组，包含最终的平均指标。
    """
    total_sums = errors_sum_array
    total_valid_pixels = total_sums[7]
    
    if total_valid_pixels == 0:
        return np.zeros(7)
    
    # [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
    final_metrics = np.zeros(7)
    
    final_metrics[0] = total_sums[0] / total_valid_pixels # abs_rel
    final_metrics[1] = total_sums[1] / total_valid_pixels # sq_rel
    final_metrics[2] = np.sqrt(total_sums[2] / total_valid_pixels) # rmse
    final_metrics[3] = np.sqrt(total_sums[3] / total_valid_pixels) # rmse_log (修正后)
    final_metrics[4] = total_sums[4] / total_valid_pixels # a1
    final_metrics[5] = total_sums[5] / total_valid_pixels # a2
    final_metrics[6] = total_sums[6] / total_valid_pixels # a3
    
    return final_metrics

# =================================================================================
# 优化部分 3: 主循环集成
# =================================================================================

def evaluate_single_scene(pred_dir, gt_dir, scene_name, num_workers, use_median_scaling=True, batch_size=16, min_depth=1e-3, max_depth=80):
    device = torch.device("cuda")
    try:
        # 使用优化的 Dataset
        dataset = PairedNpyDataset(pred_dir, gt_dir)
        # pin_memory=True 在与 .to(device) 结合时能加速内存传输
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    except ValueError as e:
        print(f"警告: 场景 {scene_name} 初始化失败 ({e})。跳过此场景。")
        return None

    all_errors_scene = []
    pbar_desc = f"评估中: {scene_name}"
    
    for pred_batch, gt_batch in tqdm(dataloader, desc=pbar_desc, unit="batch", leave=False):
        # 1. 数据传输到 GPU (得益于 pin_memory=True，速度更快)
        pred_batch, gt_batch = pred_batch.to(device), gt_batch.to(device)
        
        # <<< 优化 1: 在 GPU 上进行插值 >>>
        # 以 gt_batch 的形状为准
        target_shape = gt_batch.shape[2:] # (H, W)
        if pred_batch.shape[2:] != target_shape:
            # (B, H, W) -> (B, 1, H, W) 以便 F.interpolate
            pred_batch = pred_batch.unsqueeze(1) 
            pred_batch = F.interpolate(pred_batch, size=target_shape, mode='bilinear', align_corners=False)
            # (B, 1, H, W) -> (B, H, W)
            pred_batch = pred_batch.squeeze(1)
        
        # 2. 创建掩码
        mask_batch = (gt_batch > min_depth) & (gt_batch < max_depth) # 使用配置的min/max
        
        if use_median_scaling:
            # <<< 优化 2: 向量化的中位数定标 (移除 for 循环) >>>
            B = pred_batch.shape[0]
            
            # 1. 创建 nan 副本 (B, H, W)
            gt_nan = gt_batch.clone()
            pred_nan = pred_batch.clone()
            
            # 2. 屏蔽无效像素 (B, H, W)
            gt_nan[~mask_batch] = torch.nan
            pred_nan[~mask_batch] = torch.nan
            
            # 3. 计算每张图像的中位数 (B,)
            # dim=[1, 2] 表示在 H 和 W 维度上计算
            median_gt = torch.nanmedian(gt_nan, dim=[1, 2])
            median_pred = torch.nanmedian(pred_nan, dim=[1, 2])
            
            # 4. 计算比例 (B,)
            ratio = median_gt / median_pred
            
            # 5. 处理无效比例 (例如，全黑图像)
            # 如果 nanmedian 产生 nan (全nan图像), inf (pred 中位数为0),
            # 我们将比例设为 1.0 (即不缩放该图像)
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            
            # 6. 广播缩放 (B, H, W) * (B, 1, 1) -> (B, H, W)
            # ratio.view(B, 1, 1) 将 (B,) 变为 (B, 1, 1)
            pred_batch = pred_batch * ratio.view(B, 1, 1)
                
        # 3. 裁剪预测值 (注意: 原始脚本在定标后才裁剪，我们保持一致)
        pred_batch.clamp_(min=min_depth, max=max_depth)
        
        # <<< 优化 3: 调用新的全向量化误差函数 >>>
        # compute_errors_torch 现在在 (B, H, W) 上并行计算
        errors_batch = compute_errors_torch(gt_batch, pred_batch, mask_batch)
        
        # 4. 收集结果 (转移回CPU)
        all_errors_scene.append(errors_batch.cpu().numpy())
            
    return np.concatenate(all_errors_scene, axis=0) if all_errors_scene else None

def format_results(mean_errors, name):
    header = (" {:<65} |" + " {:>8} |" * 7).format(name, "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    line = ("-" * 67 + "|") + ("-" * 10 + "|") * 7
    values = (" {:<65} |" + "&{: 8.3f} |" * 7).format(name, *mean_errors.tolist())
    return f"{header}\n{line}\n{values}\n"

# =================================================================================
# 主函数 (无需修改)
# =================================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA GPU。程序终止。")
        exit()
        
    results_per_scene = {}
    all_errors_combined = []
    print(f"开始进行多场景评估... 模式: {'有中位数定标' if USE_MEDIAN_SCALING else '无定标 (绝对尺度)'}")
    print(f"将评估 {len(EVALUATION_PATHS)} 个自定义路径对。")
    print("-" * 50)

    for path_info in tqdm(EVALUATION_PATHS, desc="总进度", unit="scene"):
        scene_name = path_info['name']
        pred_dir = path_info['pred_dir']
        gt_dir = path_info['gt_dir']

        print(f"\n正在处理场景: {scene_name}")
        print(f" -> PRED路径: {pred_dir}")
        print(f" -> GT路径:   {gt_dir}")
        
        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            print(f"警告: 找不到场景 '{scene_name}' 的路径，已跳过。")
            continue

        scene_errors = evaluate_single_scene(
            pred_dir, gt_dir, 
            scene_name,
            num_workers=NUM_WORKERS,
            use_median_scaling=USE_MEDIAN_SCALING,
            batch_size=BATCH_SIZE,
            min_depth=MIN_EVAL_DEPTH,
            max_depth=MAX_EVAL_DEPTH
        )
        if scene_errors is not None:
            # scene_errors 是 (N, 8) 数组
            results_per_scene[scene_name] = scene_errors
            all_errors_combined.append(scene_errors)

    # ... 报告生成逻辑 (无需修改) ...
    report_lines = []
    report_lines.append("=" * 145)
    report_lines.append(f"深度评估报告 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"评估模式: {'有中位数定标 (相对几何精度)' if USE_MEDIAN_SCALING else '无定标 (绝对米制尺度)'}")
    report_lines.append(f"评估方法: 像素加权平均 (Pixel-Weighted Average)")
    report_lines.append("=" * 145)
    
    sorted_scene_names = sorted(results_per_scene.keys())
    
    for scene_name in sorted_scene_names:
        errors = results_per_scene[scene_name]
        scene_sums = errors.sum(axis=0)
        scene_mean_metrics = compute_metrics_from_sums(scene_sums)
        report_lines.append(format_results(scene_mean_metrics, scene_name))
        
    if all_errors_combined:
        all_errors_combined = np.concatenate(all_errors_combined, axis=0)
        overall_sums = all_errors_combined.sum(axis=0)
        overall_mean_metrics = compute_metrics_from_sums(overall_sums)
        
        report_lines.append("=" * 145)
        report_lines.append(format_results(overall_mean_metrics, "===> 总平均 (Overall Average)"))
        report_lines.append("=" * 145)
        
    final_report = "\n".join(report_lines)
    print("\n\n" + final_report)
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(final_report)
        
    print(f"\n报告已成功保存到: {OUTPUT_FILENAME}")
    print("评估完成!")


    # python /home/data1/szq/Megadepth/benchemarkdata/Acodes/h-evaluate-speed-all-val-hav_smbu_sztu_vis_pixel.py