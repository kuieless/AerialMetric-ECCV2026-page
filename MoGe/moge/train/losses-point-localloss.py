from typing import *
import math

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    normalized_view_plane_uv,
    angle_diff_vec3
)
from ..utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
)


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


import torch
import torch.nn.functional as F

def affine_invariant_global_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    align_resolution: int = 64, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device = pred_points.device

    # [基础 Mask]：过滤掉 Inf/NaN 和 0 值
    mask = torch.isfinite(gt_points).all(dim=-1) & (gt_points[..., 2] > 1e-6)
    
    # --- 🛡️ [修改 1] 全局熔断机制 ---
    # 如果整个 Batch 的有效像素加起来都不到 100 个，这轮训练毫无意义，直接跳过
    total_valid_pixels = mask.sum()
    if total_valid_pixels < 100:
        # 返回一个带梯度的 0 (dummy loss)，骗过 DDP，但不更新权重
        return (pred_points * 0.0).sum(), {
            'truncated_error': 0.0, 
            'delta': 0.0
        }, torch.zeros((pred_points.shape[0],), device=device)
    # -----------------------------------

    gt_points = torch.where(mask[..., None], gt_points, torch.ones_like(gt_points))

    # Align (对齐)
    # 注意：masked_nearest_resize 内部如果 mask 全 false 可能会返回空，这里通常比较安全
    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(
        pred_points, gt_points, mask=mask, size=(align_resolution, align_resolution)
    )
    
    # 计算 Scale 和 Shift
    scale, shift = align_points_scale_z_shift(
        pred_points_lr.flatten(-3, -2), 
        gt_points_lr.flatten(-3, -2), 
        lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), 
        trunc=trunc
    )
    
    valid = scale > 0
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid[..., None], shift, torch.zeros_like(shift))

    # 应用对齐
    pred_points_aligned = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

    # --- 🛡️ [修改 2] 安全的权重计算 (手写替代 weighted_mean) ---
    # 原始代码: weight = ... / gt_points
    weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
    
    # 原始风险点: weight.clamp_max(10.0 * weighted_mean(...)) -> weighted_mean 可能会除以0
    # 手动计算均值，分母加 epsilon
    weight_sum = (weight * mask.float()).sum(dim=(-2, -1), keepdim=True)
    mask_sum = mask.float().sum(dim=(-2, -1), keepdim=True)
    weight_mean = weight_sum / (mask_sum + 1e-8)  # <--- 加了 1e-8 防止除以0
    
    weight = weight.clamp_max(10.0 * weight_mean)
    # -----------------------------------

    # 计算逐像素 Loss
    # abs diff
    diff = (pred_points_aligned - gt_points).abs() * weight[..., None]
    
    # Smooth L1 (beta=0 时退化为 L1)
    if beta > 0:
        loss_map = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction='none')
    else:
        loss_map = diff

    # --- 🛡️ [修改 3] 安全的 Loss 聚合 ---
    # 原始代码: loss = ... .mean(dim=(-3, -2, -1))
    # 风险: 如果某张图 mask 全空，mean() 结果是 NaN
    
    # 我们改为只对 valid 区域求平均
    loss_sum = loss_map.sum(dim=(-3, -2, -1)) # Sum over H, W, C
    # 分母是每张图的有效像素数 * 3 (channels)
    valid_count_per_img = mask.float().sum(dim=(-2, -1)) * 3.0
    
    # 计算每张图的 Loss，如果该图无效，则 Loss=0 (利用 where)
    loss_per_img = torch.where(
        valid_count_per_img > 0,
        loss_sum / (valid_count_per_img + 1e-8),
        torch.zeros_like(loss_sum)
    )

    if sparsity_aware:
        sparsity = mask.float().mean(dim=(-2, -1)) / (lr_mask.float().mean(dim=(-2, -1)) + 1e-8)
        loss_per_img = loss_per_img / (sparsity + 1e-7)

    # 最终 Loss 取 Batch 平均 (再次过滤掉潜在的 NaN)
    # 只要 valid 的图参与计算
    valid_imgs = (valid_count_per_img > 0).float()
    final_loss = (loss_per_img * valid_imgs).sum() / (valid_imgs.sum() + 1e-8)

    # Metric 计算 (仅供参考，加上安全保护)
    with torch.no_grad():
        err = (pred_points_aligned - gt_points).norm(dim=-1) / gt_points[..., 2].clamp_min(1e-5)
        # 同样加 1e-8
        misc = {
            'truncated_error': (err.clamp_max(1.0) * mask).sum() / (mask.sum() + 1e-8),
            'delta': ((err < 1).float() * mask).sum() / (mask.sum() + 1e-8)
        }
        # 转为 python float
        misc = {k: v.item() for k, v in misc.items()}

    return final_loss, misc, scale.detach()

def monitoring(points: torch.Tensor):
    return {
        'std': points.std().item(),
    }


def compute_anchor_sampling_weight(
    points: torch.Tensor, 
    mask: torch.Tensor, 
    radius_2d: torch.Tensor, 
    radius_3d: torch.Tensor, 
    num_test: int = 64
) -> torch.Tensor:
    # Importance sampling to balance the sampled probability of fine strutures.
    # NOTE: MoGe-1 uses uniform random sampling instead of importance sampling.
    #       This is an incremental trick introduced later than the publication of MoGe-1 paper.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device), 
        torch.arange(width, device=points.device),
        indexing='ij'
    )
    
    test_delta_i = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_delta_j = torch.randint(-radius_2d, radius_2d + 1, (height, width, num_test,), device=points.device)   # [num_test]
    test_i, test_j = pixel_i[..., None] + test_delta_i, pixel_j[..., None] + test_delta_j                       # [height, width, num_test]
    test_mask = (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)                            # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(0, width - 1)                                    # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]                                                           # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]                                                                # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)                                               # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)                                          # [..., height, width]
    return weight




#     return loss, misc
import torch
import torch.nn.functional as F
import math
from typing import *

# ----------------------------------------------------------------------------
# 🔥 [修改版] 省显存的 Local Loss (稀疏采样版)
# ----------------------------------------------------------------------------
def affine_invariant_local_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    focal: torch.Tensor = None, 
    global_scale: torch.Tensor = None, 
    level: int = 4, 
    align_resolution: int = 12, 
    sample_count: int = 2048, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False,
    **kwargs
):
    """
    [Bulletproof Version] NaN-safe Local Loss.
    Fixes the 'Inf * 0 = NaN' issue and handles broadcasting robustly.
    """
    batch_size, height, width, _ = pred_points.shape
    device = pred_points.device

    # --- 🛡️ 防爆保护 1: 输入端截断 ---
    # 防止预测值出现 Inf 导致 grid_sample 采样出 Inf
    pred_points = torch.clamp(pred_points, -3000, 3000) 

    # 1. 基础 Mask
    mask = torch.isfinite(gt_points).all(dim=-1) & torch.isfinite(pred_points).all(dim=-1)
    flat_mask = mask.view(batch_size, -1) 
    
    # 2. 准备 Grid
    patch_size = 9 
    radius = align_resolution 
    step = torch.linspace(-1, 1, patch_size, device=device)
    grid_y, grid_x = torch.meshgrid(step, step, indexing='ij') 
    # [1, 81, 2]
    base_grid = torch.stack([grid_x, grid_y], dim=-1).flatten(0, 1).unsqueeze(0)

    sampled_preds = []
    sampled_gts = []
    
    for b in range(batch_size):
        valid_indices = torch.nonzero(flat_mask[b]).squeeze(1)
        num_valid = valid_indices.numel()
        
        if num_valid < 100: continue

        if num_valid > sample_count:
            perm = torch.randperm(num_valid, device=device)[:sample_count]
            indices = valid_indices[perm]
        else:
            indices = valid_indices[torch.randint(0, num_valid, (sample_count,), device=device)]
        
        # 计算采样中心
        y = (indices // width).float()
        x = (indices % width).float()
        center_x = (x / (width - 1)) * 2 - 1
        center_y = (y / (height - 1)) * 2 - 1
        
        # [sample_count, 1, 2]
        centers = torch.stack([center_x, center_y], dim=-1).view(sample_count, 1, 2)
        
        rad_x = radius / width * 2
        rad_y = radius / height * 2
        
        # [sample_count, 81, 2]
        sample_grid = centers + base_grid * torch.tensor([rad_x, rad_y], device=device).view(1, 1, 2)
        
        # 准备图像: [1, 3, H, W]
        img_p = pred_points[b].permute(2, 0, 1).unsqueeze(0)
        img_g = gt_points[b].permute(2, 0, 1).unsqueeze(0)
        
        # Grid Sample
        patches_p = F.grid_sample(img_p, sample_grid.unsqueeze(0), align_corners=True, padding_mode='border')
        patches_g = F.grid_sample(img_g, sample_grid.unsqueeze(0), align_corners=True, padding_mode='border')
        
        # 转置为 [sample_count, 81, 3]
        sampled_preds.append(patches_p.squeeze(0).permute(1, 2, 0)) 
        sampled_gts.append(patches_g.squeeze(0).permute(1, 2, 0))   

    if len(sampled_preds) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    # 合并 Batch: [Total_N, 81, 3]
    sampled_preds = torch.cat(sampled_preds, dim=0)
    sampled_gts = torch.cat(sampled_gts, dim=0)
    
    # --- 局部对齐计算 ---
    
    # 1. 过滤无效 Patch
    valid_mask = torch.isfinite(sampled_gts).all(dim=-1) & torch.isfinite(sampled_preds).all(dim=-1)
    patch_valid_mask = valid_mask.sum(dim=1) > 3
    
    p_pred = sampled_preds[patch_valid_mask]
    p_gt = sampled_gts[patch_valid_mask]
    
    # [M, 81, 1]
    p_mask = valid_mask[patch_valid_mask].unsqueeze(-1).bool() # ⚠️ 转为 bool 方便 torch.where 使用
    
    if p_pred.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    # 2. 去中心化 (Centering)
    M = p_pred.shape[0]
    p_pred = p_pred.view(M, 81, 3)
    p_gt = p_gt.view(M, 81, 3)
    p_mask = p_mask.view(M, 81, 1)

    # --- 🛡️ 防爆保护 2: 使用 torch.where 代替乘法 ---
    # 如果 p_pred 里有 Inf，p_mask 肯定是 False
    # Inf * 0 = NaN (崩溃)
    # torch.where(False, Inf, 0) = 0 (安全)
    p_pred = torch.where(p_mask, p_pred, torch.zeros_like(p_pred))
    p_gt = torch.where(p_mask, p_gt, torch.zeros_like(p_gt))
    
    # 计算均值
    count = p_mask.float().sum(dim=1, keepdim=True)
    mean_pred = p_pred.sum(dim=1, keepdim=True) / (count + 1e-6)
    mean_gt = p_gt.sum(dim=1, keepdim=True) / (count + 1e-6)
    
    # [M, 81, 3]
    p_pred_centered = (p_pred - mean_pred) # 这里不需要再乘 mask 了，因为无效区域已经是 (0 - 0) = 0
    p_gt_centered = (p_gt - mean_gt)
    
    # 再次 mask 确保万无一失 (比如均值计算带来的微小误差)
    p_pred_centered = torch.where(p_mask, p_pred_centered, torch.zeros_like(p_pred_centered))
    p_gt_centered = torch.where(p_mask, p_gt_centered, torch.zeros_like(p_gt_centered))
    
    # 3. 计算最佳缩放 (Scale)
    dot_prod = (p_pred_centered * p_gt_centered).sum(dim=1).sum(dim=1, keepdim=True)
    sq_pred = (p_pred_centered ** 2).sum(dim=1).sum(dim=1, keepdim=True)
    
    scale = dot_prod / (sq_pred + 1e-8)
    scale = torch.clamp(scale, 0.01, 100.0) # 限制局部缩放倍数
    
    # 4. 对齐
    p_pred_aligned = p_pred_centered * scale.unsqueeze(1)
    
    # 5. 计算 Loss (L1)
    diff = (p_pred_aligned - p_gt_centered).abs().sum(dim=-1)
    
    # 同样使用 safe division
    loss_per_patch = (diff * p_mask.squeeze(-1).float()).sum(dim=1) / (count.squeeze(-1).squeeze(-1) + 1e-6)
    
    return loss_per_patch.mean(), {}
# ----------------------------------------------------------------------------
# 🔥 [必备] 多尺度梯度损失 (Multi-scale Gradient Loss)
# 作用：锐化边缘，解决 "Global Loss 导致的模糊"
# ----------------------------------------------------------------------------
def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
    if params is None: params = {}
    scales = params.get('scales', 4)
    start_scale = params.get('start_scale', 0) # 🔥 [新增] 允许跳过最小尺度
    total_loss = 0
    valid_scales = 0

    for scale in range(start_scale, scales): # 🔥 从 start_scale 开始循环
        step = 2 ** scale
        if pred.shape[2] <= step or pred.shape[3] <= step: break

        # X 方向梯度
        pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
        gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
        # Y 方向梯度
        pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
        gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

        if mask is not None:
            mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
            
            if mask_x.sum() > 0:
                loss_x = torch.abs(pred_grad_x[mask_x] - gt_grad_x[mask_x]).mean()
            else:
                loss_x = 0.0
                
            if mask_y.sum() > 0:
                loss_y = torch.abs(pred_grad_y[mask_y] - gt_grad_y[mask_y]).mean()
            else:
                loss_y = 0.0
        else:
            loss_x = torch.abs(pred_grad_x - gt_grad_x).mean()
            loss_y = torch.abs(pred_grad_y - gt_grad_y).mean()

        total_loss += (loss_x + loss_y)
        valid_scales += 1

    if valid_scales > 0:
        return total_loss / valid_scales
    return torch.tensor(0.0, device=pred.device, requires_grad=True)


def normal_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))

    return loss, {}


def edge_loss(points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    device, dtype = points.device, points.dtype
    height, width = points.shape[-3:-1]

    mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(mask[..., None], gt_points, 1)

    dx = points[..., :-1, :, :] - points[..., 1:, :, :]
    dy = points[..., :, :-1, :] - points[..., :, 1:, :]
    
    gt_dx = gt_points[..., :-1, :, :] - gt_points[..., 1:, :, :]
    gt_dy = gt_points[..., :, :-1, :] - gt_points[..., :, 1:, :]

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * _smooth(angle_diff_vec3(dx, gt_dx).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss_dy = mask_dy * _smooth(angle_diff_vec3(dy, gt_dy).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)
    loss = (loss_dx.mean(dim=(-2, -1)) + loss_dy.mean(dim=(-2, -1))) / (2 * max(points.shape[-3:-1]))

    return loss, {}


def mask_l2_loss(pred_mask: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = gt_mask_neg.float() * pred_mask.square() + gt_mask_pos.float() * (1 - pred_mask).square()
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean(dim=(-2, -1))
    return loss, {}


# def metric_scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor):
#     valid = scale_gt > 0
#     return torch.where(valid, F.mse_loss(scale_pred.log(), torch.where(valid, scale_gt.log(), 0), reduction='none'), 0), {}
def metric_scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor, mask=None):
    # 兼容处理：如果没传 mask，就根据 gt > 0 自动生成
    if mask is None:
        valid = scale_gt > 1e-6
    else:
        valid = mask & (scale_gt > 1e-6)

    # --- 🛡️ 保护 1: 如果全是无效点，直接返回 0 ---
    if valid.sum() < 1:
        # 返回一个带梯度的标量 0
        return (scale_pred * 0.0).sum(), {}

    # 计算 Log
    log_pred = scale_pred.log()
    # 使用 where 避免对 0 取 log 产生 NaN
    log_gt = torch.where(valid, scale_gt.log(), torch.zeros_like(scale_gt))

    # 计算 MSE (保留 reduction='none' 以便后续手动过滤)
    loss_map = F.mse_loss(log_pred, log_gt, reduction='none')

    # --- 🛡️ 保护 2: 手动计算平均值 (返回标量) ---
    # 只对 valid 的区域求和，然后除以有效点数量
    loss = (loss_map * valid.float()).sum() / (valid.float().sum() + 1e-8)
    return loss, {}


def focal_frequency_loss(pred_points, gt_points, mask=None):
    """
    在频域计算损失，强迫模型学习高频细节（纹理、边缘）。
    注意：MoGe 输出的是 3D Points，我们主要对 Depth (Z轴) 进行频域约束。
    """
    import torch.fft

    # 1. 提取深度 (Z轴)
    pred_depth = pred_points[..., 2]
    gt_depth = gt_points[..., 2]

    # 2. 基础 Mask 处理 (NaN 补 0，防止 FFT 炸裂)
    # 如果传入了 mask，利用 mask；如果没有，根据 gt 有效性生成
    if mask is None:
        mask = torch.isfinite(gt_points).all(dim=-1)
    
    # 将无效区域填 0 (或者填均值，填 0 最安全)
    pred_filled = torch.where(mask, pred_depth, torch.zeros_like(pred_depth))
    gt_filled = torch.where(mask, gt_depth, torch.zeros_like(gt_depth))

    # 3. 转换到频域 (2D FFT)
    # 维度: [Batch, H, W] -> [Batch, H, W] (复数)
    pred_freq = torch.fft.fft2(pred_filled, norm='ortho')
    gt_freq = torch.fft.fft2(gt_filled, norm='ortho')

    # 4. 计算频域差异 (幅度谱的平方差)
    # 使用 real^2 + imag^2
    diff = torch.abs(pred_freq - gt_freq) ** 2

    # 5. 动态加权 (Focal)
    # 这一步是核心：对于差异大的频率（通常是高频纹理），给予更大的权重
    weight = diff / (diff.mean(dim=(-2, -1), keepdim=True) + 1e-8)
    
    # 6. 计算 Loss
    loss_map = weight * diff
    loss = loss_map.mean() # 频域整体求均值

    return loss, {}




def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_normal).all(dim=-1)
    gt_normal = torch.where(mask[..., None], gt_normal, 1)

    loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
    return loss, {}





def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
    """
    多尺度梯度损失：强迫预测图的边缘（梯度）与真值一致。
    这对去模糊、恢复高频细节极其重要。
    """
    if params is None: params = {}
    scales = params.get('scales', 4)
    
    total_loss = 0
    valid_scales = 0

    for scale in range(scales):
        step = 2 ** scale
        
        # 如果缩放后尺寸不够，就停止
        if pred.shape[2] <= step or pred.shape[3] <= step:
            break

        # 1. 计算 X 方向梯度 (像素 i 和 i+step 的差)
        pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
        gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
        
        # 2. 计算 Y 方向梯度
        pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
        gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

        # 3. 处理 Mask (Mask 也要对应的切片)
        if mask is not None:
            mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
            
            # 只在有效区域计算 L1 Loss
            loss_x = (torch.abs(pred_grad_x - gt_grad_x) * mask_x).sum() / (mask_x.sum() + 1e-8)
            loss_y = (torch.abs(pred_grad_y - gt_grad_y) * mask_y).sum() / (mask_y.sum() + 1e-8)
        else:
            loss_x = torch.abs(pred_grad_x - gt_grad_x).mean()
            loss_y = torch.abs(pred_grad_y - gt_grad_y).mean()

        total_loss += (loss_x + loss_y)
        valid_scales += 1

    if valid_scales > 0:
        return total_loss / valid_scales, {}
    
    # 避免返回 None 导致报错
    return torch.tensor(0.0, device=pred.device, requires_grad=True), {}

# ==========================================
# Append this to the end of moge/train/losses.py
# ==========================================

def local_planar_loss(pred_depth, gt_depth, mask=None):
    """
    航拍专属：局部平面约束 (去噪神器)。
    强迫平坦区域的二阶导数接近0，消除 Gradient Loss 带来的波浪纹。
    """
    if mask is None:
        mask = torch.isfinite(gt_depth) & torch.isfinite(pred_depth)
        
    # 使用 Log Depth 保证数值稳定性
    # clamp 防止 log(0)
    log_pred = torch.log(pred_depth.clamp(min=1e-6))
    
    # 手动计算 Laplacian (二阶导数)
    # Center: [B, H-2, W-2]
    d_center = log_pred[..., 1:-1, 1:-1]
    
    # Neighbors: 上下左右
    d_sum = log_pred[..., :-2, 1:-1] + \
            log_pred[..., 2:,  1:-1] + \
            log_pred[..., 1:-1, :-2] + \
            log_pred[..., 1:-1, 2:]
            
    # Laplacian = 4*Center - Sum(Neighbors)
    # 如果是完全平坦或线性变化的平面，这个值理论上为 0
    laplacian = 4 * d_center - d_sum
    
    # --- 边缘感知权重 (Edge-Aware Weighting) ---
    # 我们只希望在“平坦区域”去噪，不要把“建筑边缘”给磨平了
    # 计算 GT 的梯度，梯度大的地方权重设为 0
    gt_log = torch.log(gt_depth.clamp(min=1e-6))
    gt_dy = torch.abs(gt_log[..., 2:, 1:-1] - gt_log[..., :-2, 1:-1])
    gt_dx = torch.abs(gt_log[..., 1:-1, 2:] - gt_log[..., 1:-1, :-2])
    
    # exp(-grad): GT 梯度越大(边缘)，权重越接近 0；平坦区域权重接近 1
    smooth_weight = torch.exp(-(gt_dx + gt_dy))
    
    # Mask 处理 (去掉边缘 1 像素)
    mask_center = mask[..., 1:-1, 1:-1]
    
    # Loss 计算
    loss_map = torch.abs(laplacian) * smooth_weight
    
    # 只统计有效区域
    if mask_center.sum() > 0:
        loss = (loss_map * mask_center.float()).sum() / (mask_center.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
        
    return loss, {}

# ==========================================
# Append this to the end of moge/train/losses.py
# ==========================================

# ==========================================
# 替换 moge/train/losses.py 中的 geometry_consistency_loss
# ==========================================

def geometry_consistency_loss(pred_points, gt_points, mask=None):
    """
    几何一致性 Loss (法向量 Loss)。
    修复了切片维度错误，确保是在 Spatial 维度差分，保留 Channel=3。
    """
    def compute_normal(points):
        # points shape: [B, H, W, 3]
        
        # 1. 计算 X 方向梯度 (Width 维度，倒数第2维)
        # points[..., :, 2:, :] 表示: Batch, Height, Width(2:end), Channel(all)
        vec_x = points[..., :, 2:, :] - points[..., :, :-2, :]
        
        # 2. 计算 Y 方向梯度 (Height 维度，倒数第3维)
        # points[..., 2:, :, :] 表示: Batch, Height(2:end), Width(all), Channel(all)
        vec_y = points[..., 2:, :, :] - points[..., :-2, :, :]
        
        # 3. 对齐尺寸
        # vec_x 当前是 [B, H, W-2, 3]，我们需要裁掉 H 的头尾
        vec_x = vec_x[..., 1:-1, :, :] 
        
        # vec_y 当前是 [B, H-2, W, 3]，我们需要裁掉 W 的头尾
        vec_y = vec_y[..., :, 1:-1, :]
        
        # 现在两者都是 [B, H-2, W-2, 3]，可以在最后一维做叉乘了
        normal = torch.cross(vec_x, vec_y, dim=-1)
        normal = F.normalize(normal, dim=-1, p=2)
        return normal

    # 2. 计算 Pred 和 GT 的法向量
    pred_norm = compute_normal(pred_points)
    gt_norm = compute_normal(gt_points)
    
    # 3. 计算 Cosine Similarity Loss
    # 维度对齐后，Map 尺寸会比原图小 2 像素
    cos_sim = (pred_norm * gt_norm).sum(dim=-1)
    loss_map = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)
    
    # 4. Mask 处理
    if mask is None:
        mask = torch.isfinite(gt_points).all(dim=-1) & torch.isfinite(pred_points).all(dim=-1)
    
    # Mask 也要相应的裁剪 1 像素边缘
    mask_valid = mask[..., 1:-1, 1:-1]
    
    if mask_valid.sum() > 0:
        return (loss_map * mask_valid.float()).sum() / (mask_valid.sum() + 1e-8), {}
    else:
        return torch.tensor(0.0, device=pred_points.device, requires_grad=True), {}