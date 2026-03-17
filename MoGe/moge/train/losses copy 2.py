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


def affine_invariant_local_loss(
    pred_points: torch.Tensor, 
    gt_points: torch.Tensor, 
    focal: torch.Tensor, 
    global_scale: torch.Tensor, 
    level: Literal[4, 16, 64], 
    align_resolution: int = 32, 
    num_patches: int = 16, 
    beta: float = 0.0, 
    trunc: float = 1.0, 
    sparsity_aware: bool = False
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)

    gt_mask = torch.isfinite(gt_points).all(dim=-1)
    gt_points = torch.where(gt_mask[..., None], gt_points, 1)
    pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
    radius_3d = 0.5 / level / focal * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
    where_mask = torch.where(gt_mask)
    random_selection = torch.multinomial(anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True)
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]     # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device), 
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij'
    )
    patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)
    
    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)    
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}
    
    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = patch_batch_idx[nonempty], patch_i[nonempty], patch_j[nonempty]
    patch_mask = patch_mask[nonempty]                                   # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]                         # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]                   # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]           # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    
    # Align patch points
    pred_patch_points_lr, gt_patch_points_lr, patch_lr_mask = utils3d.pt.masked_nearest_resize(pred_patch_points, gt_patch_points, mask=patch_mask, size=(align_resolution, align_resolution))
    local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc)
    if global_scale is not None:
        scale_differ = local_scale / global_scale[patch_batch_idx]
        patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale > 0)
    else:
        patch_valid = local_scale > 0
    local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    patch_mask &= patch_valid[:, None, None]

    pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                   # [num_patches_nonempty, patch_h, patch_w, 3]
    
    # Compute loss
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])          # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth((pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=(-3, -2, -1))    # [num_patches_nonempty]
    
    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    loss = torch.scatter_reduce(torch.zeros(batch_size, dtype=dtype, device=device), dim=0, index=patch_batch_idx, src=loss, reduce='sum') / num_patches
    loss = loss.reshape(batch_shape)
    
    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[..., None, None]

    # Record any scalar metric
    misc = {
        'truncated_error': weighted_mean(err.clamp_max(1), patch_mask).item(),
        'delta': weighted_mean((err < 1).float(), patch_mask).item()
    }

    return loss, misc


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

# ==========================================
# 🔥 魔法二：SiLog Loss (Scale-Invariant Log)
# 专治：绝对尺度预测不准，远近误差不平衡
# ==========================================
# def metric_scale_loss(scale_pred, scale_gt, mask=None, params=None):
#     """
#     替代原本简单的 metric_scale_loss。
#     params: {'variance_focus': 0.85} 
#     variance_focus: 0.85 是推荐值，兼顾相对关系和绝对数值。
#     """
#     if params is None: params = {}
#     variance_focus = params.get('variance_focus', 0.85)

#     # 1. Mask 处理
#     if mask is None:
#         valid = scale_gt > 1e-6
#     else:
#         valid = mask & (scale_gt > 1e-6)

#     if valid.sum() < 1:
#         return (scale_pred * 0.0).sum(), {}

#     # 2. 计算 Log 差值 (d)
#     # 加上 clamp 防止 log(0)
#     log_pred = torch.log(torch.clamp(scale_pred, min=1e-6))
#     log_gt = torch.log(torch.clamp(scale_gt, min=1e-6))
    
#     d = log_pred - log_gt
    
#     # 只取有效值
#     d = d[valid]

#     # 3. SiLog 公式
#     # Loss = sqrt( mean(d^2) - lambda * mean(d)^2 )
#     # 第一项 (d**2).mean(): 使得绝对误差变小
#     # 第二项 (d.mean())**2: 使得整体尺度的偏差变小
    
#     diff_sq_mean = (d ** 2).mean()
#     diff_mean_sq = (d.mean()) ** 2
    
#     loss = torch.sqrt(diff_sq_mean - variance_focus * diff_mean_sq + 1e-8)
    
#     # 4. 额外加一个 L1 Log 惩罚，防止绝对值飘太远
#     abs_log_loss = torch.abs(d).mean()
    
#     # 组合: SiLog + 0.1 * AbsLog
#     return loss + 0.1 * abs_log_loss, {}


def normal_map_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(gt_normal).all(dim=-1)
    gt_normal = torch.where(mask[..., None], gt_normal, 1)

    loss = (mask * utils3d.pt.angle_between(pred_normal, gt_normal).square()).mean(dim=(-2, -1))
    return loss, {}



import torch
import torch.nn.functional as F

# ==========================================
# 🔥 新增：多尺度梯度损失 (修复模糊神器)
# ==========================================
def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
    """
    计算多尺度的梯度损失，强迫预测的边缘（梯度）与真值一致。
    """
    if params is None: params = {}
    scales = params.get('scales', 4)
    
    total_loss = 0
    valid_scales = 0

    for scale in range(scales):
        step = 2 ** scale
        
        # 如果图片太小不够算梯度，就停止
        if pred.shape[2] <= step or pred.shape[3] <= step:
            break

        # 计算 X 方向梯度 (相邻 step 像素相减)
        pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
        gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
        
        # 计算 Y 方向梯度
        pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
        gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

        # 处理 Mask (如果有的话)
        if mask is not None:
            # Mask 也要跟着切片，保证对齐
            mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
            
            # 计算 L1 Loss (只计算有效区域)
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

        # 累加损失 (越大的尺度权重越小，或者平均)
        total_loss += (loss_x + loss_y)
        valid_scales += 1

    if valid_scales > 0:
        return total_loss / valid_scales
    return torch.tensor(0.0, device=pred.device, requires_grad=True)