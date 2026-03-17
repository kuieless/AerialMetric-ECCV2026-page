import os
from pathlib import Path
import sys
# 保持原有的路径添加逻辑
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import time
import random
from typing import *
import itertools
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import utils3d
import click
from tqdm import tqdm, trange
import mlflow
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

from moge.train.dataloader import TrainDataLoaderPipeline
from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss, 
    edge_loss,
    normal_loss, 
    mask_l2_loss, 
    mask_bce_loss,
    metric_scale_loss,
    normal_map_loss,
    monitoring, 
)
from moge.train.utils import build_optimizer, build_lr_scheduler
from moge.utils.geometry_torch import intrinsics_to_fov
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.tools import key_average, recursive_replace, CallbackOnException, flatten_nested_dict
from moge.test.metrics import compute_metrics


def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
    if params is None: params = {}
    scales = params.get('scales', 4)
    total_loss = 0
    valid_scales = 0

    for scale in range(scales):
        step = 2 ** scale
        if pred.shape[2] <= step or pred.shape[3] <= step: break

        # Calculate gradients
        pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
        gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
        pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
        gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

        # Create a validity mask (1 where data is good, 0 where data is NaN/Inf)
        valid_mask_x = torch.isfinite(gt_grad_x) & torch.isfinite(pred_grad_x)
        valid_mask_y = torch.isfinite(gt_grad_y) & torch.isfinite(pred_grad_y)

        # Combine with user-provided mask if it exists
        if mask is not None:
            mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
            valid_mask_x = valid_mask_x & mask_x
            valid_mask_y = valid_mask_y & mask_y

        # Compute Loss only on valid pixels
        if valid_mask_x.sum() > 0:
            diff_x = torch.abs(pred_grad_x - gt_grad_x)
            loss_x = diff_x[valid_mask_x].mean()
        else:
            loss_x = 0.0

        if valid_mask_y.sum() > 0:
            diff_y = torch.abs(pred_grad_y - gt_grad_y)
            loss_y = diff_y[valid_mask_y].mean()
        else:
            loss_y = 0.0

        total_loss += (loss_x + loss_y)
        valid_scales += 1

    if valid_scales > 0:
        return total_loss / valid_scales
    return torch.tensor(0.0, device=pred.device, requires_grad=True)

@click.command()
@click.option('--config', 'config_path', type=str, default='configs/debug.json')
@click.option('--workspace', type=str, default='workspace/debug', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load. "latest" to load latest checkpoint in workspace, integer to load by step number')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16/BF16')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size')
@click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
@click.option('--seed', type=int, default=0, help='Random seed')
@click.option('--debug_anomaly', type=bool, default=False, help='Enable PyTorch Anomaly Detection')
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    enable_mixed_precision: bool,
    enable_ema: bool,
    num_iterations: int,
    save_every: int,
    log_every: int,
    vis_every: int,
    num_vis_images: int,
    enable_mlflow: bool,
    seed: Optional[int],
    debug_anomaly: bool,
):
    if debug_anomaly:
        print("🚨 [DEBUG MODE] torch.autograd.set_detect_anomaly(True) is ENABLED.")
        torch.autograd.set_detect_anomaly(True)

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 自动适配精度
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=None, 
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    device = accelerator.device
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

    # Log config
    if accelerator.is_main_process:
        if enable_mlflow:
            try:
                mlflow.log_params({**click.get_current_context().params, 'batch_size_total': batch_size_total})
            except:
                print('Failed to log config to MLFlow')
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)

    if seed is not None: 
        set_seed(seed, device_specific=True)

    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(config['model_version'])      
        model = MoGeModel(**config['model'])

    # ================= 🛡️ FSDP 策略 =================
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        try:
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
                block_module = model.encoder.backbone.blocks[0]
                block_class = type(block_module)
                import functools
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_class})
                accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
        except Exception as e:
            pass

    # ================= 🟢 全参数微调设置 🟢 =================
    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'🔥 Total parameters: {count_total_parameters}')
    
    # 确保所有层 requires_grad=True（解冻 Backbone）
    for param in model.parameters():
        param.requires_grad = True

    count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'🔥 Trainable parameters: {count_trainable} (Ratio: {count_trainable/count_total_parameters:.2%})')

    # 开启训练模式
    model.train()

    # 🛡️ 极其重要的护身符：精确冻结 BN，放开 LN！
    # 因为单卡 Batch Size 过小（如 8），更新 BatchNorm 的均值和方差会引发训练崩塌。
    # 但由于你是全参数微调，ViT 内部的 LayerNorm 必须允许更新。
    def freeze_bn_stats(m):
        # 仅冻结受 Batch Size 影响的标准 BatchNorm。不能冻结 LayerNorm！
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            m.eval() 
    
    print("🛡️ [Safe Mode] Freezing BN statistics (keeping LayerNorm active for ViT)...")
    model.apply(freeze_bn_stats)
    # ===========================================================================

    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    # Attempt to load checkpoint
    checkpoint: Dict[str, Any]
    with accelerator.local_main_process_first():
        if checkpoint_path is None:
            checkpoint = None
        elif checkpoint_path.endswith('.pt'):
            print(f'Load checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        elif checkpoint_path == "latest": 
            checkpoint_path = Path(workspace, 'checkpoint', 'latest.pt')
            if checkpoint_path.exists():
                print(f'Load checkpoint: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                i_step = checkpoint['step']
                if 'model' not in checkpoint and (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                    checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
                if 'optimizer' not in checkpoint and (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
                    checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
                if enable_ema and accelerator.is_main_process:
                    if 'ema_model' not in checkpoint and (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
                        checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
            else:
                print(f'No latest checkpoint found. Start from scratch.')
                checkpoint = None
        else:
            i_step = int(checkpoint_path)
            checkpoint = {'step': i_step}
            if (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
            if (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
                checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
            if enable_ema and accelerator.is_main_process:
                if (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
                    checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']

    if checkpoint is None:
        print('Initialize model weights')
        with accelerator.local_main_process_first():
            model.init_weights()
        initial_step = 0
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'step' in checkpoint:
            initial_step = checkpoint['step'] + 1
        else:
            initial_step = 0
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
            ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        del checkpoint
    
    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    with accelerator.local_main_process_first():
        train_data_pipe = TrainDataLoaderPipeline(config['data'], batch_size_forward, num_load_workers=6, num_process_workers=6, buffer_size=32)

    def _write_bytes_retry_loop(save_path: Path, data: bytes):
        while True:
            try:
                save_path.write_bytes(data)
                break
            except Exception as e:
                print('Error while saving checkpoint, retrying in 1 minute: ', e)
                time.sleep(60)

    records = []
    
    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):  
        if accelerator.is_main_process:
            batches_for_vis: List[Dict[str, torch.Tensor]] = []
            num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
            for _ in range(num_vis_images // batch_size_forward):
                batch = train_data_pipe.get()
                batches_for_vis.append(batch)

        if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
            save_dir = Path(workspace).joinpath('vis/gt')
            for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
                image, gt_depth, gt_normal, gt_intrinsics, info = batch['image'], batch['depth'], batch['normal'], batch['intrinsics'], batch['info']
                gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                for i_instance in range(batch['image'].shape[0]):
                    idx = i_batch * batch_size_forward + i_instance
                    image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_depth_i = gt_depth[i_instance].numpy()
                    gt_points_i = gt_points[i_instance].numpy()
                    gt_normal_i = gt_normal[i_instance].numpy()
                    save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(gt_points_i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(gt_depth_i), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal.png')), cv2.cvtColor(colorize_normal(gt_normal_i), cv2.COLOR_RGB2BGR))
                    with save_dir.joinpath(f'{idx:04d}/info.json').open('w') as f:
                        json.dump(info[i_instance], f)

        if seed is not None: set_seed(seed + initial_step, device_specific=True)   

        # Training loop
        for i_step in range(initial_step, num_iterations):
            
            # 确保每一轮开始时，BN 状态正确（防止被其它过程覆盖）
            model.apply(freeze_bn_stats)

            i_accumulate = 0
            while i_accumulate < gradient_accumulation_steps:
                try:
                    batch = train_data_pipe.get()
                except Exception as e:
                    print(f"⚠️ Dataloader Error: {e}")
                    continue

                image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = \
                    batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']
                
                image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = \
                    image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                
                current_batch_size = image.shape[0]
                gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                
                # 🛡️ 保护 1: Focal 除零保护
                fx, fy = gt_intrinsics[..., 0, 0].float(), gt_intrinsics[..., 1, 1].float()
                gt_focal = (1.0 / torch.sqrt(1.0 / (fx ** 2 + 1e-6) + 1.0 / (fy ** 2 + 1e-6))).float()

                with accelerator.accumulate(model):
                    num_tokens = config['model']['num_tokens_range'][0] if i_step <= config.get('low_resolution_training_steps', 0) else accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
                    autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
                    with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=accelerator.mixed_precision != 'no'):
                        output = model(image, num_tokens=num_tokens)
                    
                    pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])

                    # 🛡️ 防爆版 Loss 计算 (强制 FP32，执行样本级清洗)
                    with torch.autocast(device_type=device.type, enabled=False):
                        valid_loss_list = []
                        
                        for i in range(current_batch_size):
                            current_label = str(label_type[i])
                            if current_label not in config['loss']: continue

                            p_points = pred_points[i].float()
                            p_normal = pred_normal[i].float() if pred_normal is not None else None
                            p_mask = pred_mask[i].float() if pred_mask is not None else None
                            p_metric_scale = pred_metric_scale[i].float() if pred_metric_scale is not None else None

                            g_points, g_normal = gt_points[i].float(), gt_normal[i].float()
                            g_mask_fin, g_mask_inf, g_focal_i = gt_mask_fin[i], gt_mask_inf[i], gt_focal[i]

                            gt_metric_scale = None 
                            loss_dict, weight_dict, misc_dict = {}, {}, {}
                            is_sample_broken = False 
                            
                            for k, v in config['loss'][current_label].items():
                                weight_dict[k] = v['weight']
                                term_loss = torch.tensor(0.0, device=device)

                                try:
                                    if v['function'] == 'affine_invariant_global_loss':
                                        term_loss, misc_dict[k], gt_metric_scale = affine_invariant_global_loss(p_points, g_points, **v['params'])
                                    elif v['function'] == 'affine_invariant_local_loss':
                                        term_loss, misc_dict[k] = affine_invariant_local_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), g_focal_i, gt_metric_scale, **v['params'])
                                    elif v['function'] == 'geometry_consistency_loss':
                                        val, _ = geometry_consistency_loss(p_points, g_points)
                                        term_loss = val 
                                    elif v['function'] == 'normal_loss':
                                        term_loss, misc_dict[k] = normal_loss(p_points, g_points)
                                    elif v['function'] == 'edge_loss':
                                        term_loss, misc_dict[k] = edge_loss(p_points, g_points)
                                    elif v['function'] == 'normal_map_loss':
                                        term_loss, misc_dict[k] = normal_map_loss(p_normal, g_normal)
                                    elif v['function'] == 'mask_bce_loss':
                                        term_loss, misc_dict[k] = mask_bce_loss(p_mask, g_mask_fin, g_mask_inf)
                                    elif v['function'] == 'mask_l2_loss':
                                        term_loss, misc_dict[k] = mask_l2_loss(p_mask, g_mask_fin, g_mask_inf)
                                    elif v['function'] == 'multi_scale_gradient_loss':
                                        term_loss = multi_scale_gradient_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), params=v.get('params', {}))
                                        misc_dict[k] = term_loss
                                    elif v['function'] == 'metric_scale_loss' and is_metric[i] and p_metric_scale is not None:
                                        term_loss, misc_dict[k] = metric_scale_loss(p_metric_scale, gt_metric_scale)
                                except Exception as e:
                                    print(f"🚨 [Step {i_step}] Loss Error [{k}]: {str(e)}")
                                    is_sample_broken = True
                                    break
                                
                                loss_dict[k] = term_loss
                                
                                if torch.isnan(term_loss).item() or torch.isinf(term_loss).item():
                                    is_sample_broken = True
                                    break 
                            
                            # 若此图片包含 NaN/Inf，直接丢弃
                            if is_sample_broken: 
                                continue 

                            weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                            loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                            
                            sample_total_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                            if not isinstance(sample_total_loss, torch.Tensor):
                                sample_total_loss = torch.tensor(sample_total_loss, dtype=torch.float32, device=device)

                            if torch.isnan(sample_total_loss).item() or torch.isinf(sample_total_loss).item():
                                continue
                            
                            # 🛡️ 保护: 硬截断防爆炸。控制在最高 500
                            if sample_total_loss > 500.0:
                                sample_total_loss = torch.clamp(sample_total_loss, max=500.0)

                            valid_loss_list.append(sample_total_loss)
                            
                            # 记录 Metrics
                            safe_misc_dict = {}
                            misc_dict = flatten_nested_dict(misc_dict)
                            for k_misc, v_misc in misc_dict.items():
                                k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
                                safe_misc_dict[k_misc_str] = v_misc.detach().float().item() if isinstance(v_misc, torch.Tensor) else v_misc
                            safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                            records.append({**safe_loss_dict, **safe_misc_dict})

                        # 4. Batch 级聚合与 Backward
                        if len(valid_loss_list) > 0:
                            loss = sum(valid_loss_list) / len(valid_loss_list)
                            
                            if debug_anomaly:
                                with torch.autograd.detect_anomaly():
                                    accelerator.backward(loss)
                            else:
                                accelerator.backward(loss)

                            if accelerator.sync_gradients:
                                params_to_check = [p for p in model.parameters() if p.grad is not None]
                                if len(params_to_check) > 0:
                                    grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
                                    if torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item():
                                        if accelerator.is_main_process:
                                            pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping.")
                                        optimizer.zero_grad(set_to_none=True)
                                    else:
                                        optimizer.step()
                                        optimizer.zero_grad(set_to_none=True)
                                else:
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                        else:
                            # 防御性 Dummy Backward (防止 DDP 卡死)
                            dummy_loss = pred_points.sum() * 0.0
                            accelerator.backward(dummy_loss)
                            optimizer.zero_grad(set_to_none=True)

                i_accumulate += 1

            lr_scheduler.step()

            # EMA 更新
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                if accelerator.distributed_type == accelerate.DistributedType.FSDP:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                        ema_model.update_parameters(model)
                else:
                    ema_model.update_parameters(model)

            if i_step == initial_step or i_step % log_every == 0:
                records = [key_average(records)]
                records = accelerator.gather_for_metrics(records, use_gather_object=True)
                if accelerator.is_main_process:
                    records = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in r.items()} for r in records]
                    records = key_average(records)
                    if enable_mlflow:
                        try: mlflow.log_metrics(records, step=i_step)
                        except: pass
                records = []

            # Save checkponits
            if accelerator.is_main_process and (i_step % save_every == 0):
                Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)
                
                # 1. 保存模型权重
                with io.BytesIO() as f: 
                    torch.save({'model_config': config['model'], 'model': accelerator.unwrap_model(model).state_dict()}, f)
                    model_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), model_bytes)
                
                # 2. 保存优化器状态
                with io.BytesIO() as f: 
                    torch.save({'model_config': config['model'], 'step': i_step, 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}, f)
                    opt_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'), opt_bytes)
                
                # 3. 保存 EMA 权重 (如果开启)
                if enable_ema:
                    with io.BytesIO() as f: 
                        torch.save({'model_config': config['model'], 'model': ema_model.module.state_dict()}, f)
                        ema_bytes = f.getvalue()
                    save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'), ema_bytes)

                # 4. 保存 Latest 标记
                with io.BytesIO() as f: 
                    torch.save({'model_config': config['model'], 'step': i_step}, f)
                    latest_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), latest_bytes)
            
            # Visualize
            if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0):
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
                save_dir.mkdir(parents=True, exist_ok=True)
                with torch.inference_mode():
                    for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
                        image, gt_depth, gt_intrinsics = batch['image'], batch['depth'], batch['intrinsics']
                        image, gt_depth, gt_intrinsics = image.to(device), gt_depth.to(device), gt_intrinsics.to(device)
                        output = unwrapped_model.infer(image)
                        pred_points = output['points'].cpu().numpy() if 'points' in output else None
                        pred_depth = output['depth'].cpu().numpy() if 'depth' in output else None
                        pred_mask =  output['mask'].cpu().numpy() if 'mask' in output else None
                        pred_normal = output['normal'].cpu().numpy() if 'normal' in output else None
                        image = (image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                        for i_instance in range(image.shape[0]):
                            idx = i_batch * batch_size_forward + i_instance
                            save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image[i_instance], cv2.COLOR_RGB2BGR))
                            if pred_points is not None:
                                cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(pred_points[i_instance], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                            if pred_depth is not None:
                                cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(pred_depth[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
                            if pred_normal is not None:
                                cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal_vis.png')), cv2.cvtColor(colorize_normal(pred_normal[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
            
            pbar.set_postfix({'loss': loss.item() if isinstance(loss, torch.Tensor) else loss}, refresh=False)
            pbar.update(1)

if __name__ == '__main__':
    main()
# import os
# from pathlib import Path
# import sys
# # 保持原有的路径添加逻辑
# if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
#     sys.path.insert(0, _package_root)
# import json
# import time
# import random
# from typing import *
# import itertools
# from contextlib import nullcontext
# from concurrent.futures import ThreadPoolExecutor
# import io

# import numpy as np
# import cv2
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.version
# import accelerate
# from accelerate import Accelerator, DistributedDataParallelKwargs
# from accelerate.utils import set_seed
# import utils3d
# import click
# from tqdm import tqdm, trange
# import mlflow
# torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

# from moge.train.dataloader import TrainDataLoaderPipeline
# from moge.train.losses import (
#     affine_invariant_global_loss,
#     affine_invariant_local_loss, 
#     edge_loss,
#     normal_loss, 
#     mask_l2_loss, 
#     mask_bce_loss,
#     metric_scale_loss,
#     normal_map_loss,
#     monitoring, 
# )
# from moge.train.utils import build_optimizer, build_lr_scheduler
# from moge.utils.geometry_torch import intrinsics_to_fov
# from moge.utils.vis import colorize_depth, colorize_normal
# from moge.utils.tools import key_average, recursive_replace, CallbackOnException, flatten_nested_dict
# from moge.test.metrics import compute_metrics


# def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
#     if params is None: params = {}
#     scales = params.get('scales', 4)
#     total_loss = 0
#     valid_scales = 0

#     for scale in range(scales):
#         step = 2 ** scale
#         if pred.shape[2] <= step or pred.shape[3] <= step: break

#         # Calculate gradients
#         pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
#         gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
#         pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
#         gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

#         # Create a validity mask (1 where data is good, 0 where data is NaN/Inf)
#         # We must check both GT and Pred for validity
#         valid_mask_x = torch.isfinite(gt_grad_x) & torch.isfinite(pred_grad_x)
#         valid_mask_y = torch.isfinite(gt_grad_y) & torch.isfinite(pred_grad_y)

#         # Combine with user-provided mask if it exists
#         if mask is not None:
#             mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
#             mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
#             valid_mask_x = valid_mask_x & mask_x
#             valid_mask_y = valid_mask_y & mask_y

#         # Compute Loss only on valid pixels
#         if valid_mask_x.sum() > 0:
#             diff_x = torch.abs(pred_grad_x - gt_grad_x)
#             # Only sum errors where mask is True
#             loss_x = diff_x[valid_mask_x].mean()
#         else:
#             loss_x = 0.0

#         if valid_mask_y.sum() > 0:
#             diff_y = torch.abs(pred_grad_y - gt_grad_y)
#             loss_y = diff_y[valid_mask_y].mean()
#         else:
#             loss_y = 0.0

#         total_loss += (loss_x + loss_y)
#         valid_scales += 1

#     if valid_scales > 0:
#         return total_loss / valid_scales
#     return torch.tensor(0.0, device=pred.device, requires_grad=True)

# @click.command()
# @click.option('--config', 'config_path', type=str, default='configs/debug.json')
# @click.option('--workspace', type=str, default='workspace/debug', help='Path to the workspace')
# @click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load. "latest" to load latest checkpoint in workspace, integer to load by step number')
# @click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
# @click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
# @click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
# @click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
# @click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
# @click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
# @click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
# @click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
# @click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
# @click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size')
# @click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
# @click.option('--seed', type=int, default=0, help='Random seed')
# def main(
#     config_path: str,
#     workspace: str,
#     checkpoint_path: str,
#     batch_size_forward: int,
#     gradient_accumulation_steps: int,
#     enable_gradient_checkpointing: bool,
#     enable_mixed_precision: bool,
#     enable_ema: bool,
#     num_iterations: int,
#     save_every: int,
#     log_every: int,
#     vis_every: int,
#     num_vis_images: int,
#     enable_mlflow: bool,
#     seed: Optional[int],
# ):
#     # Load config
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     # 🔥 [Fix 1: BF16 Support] 不要写死 fp16，让 accelerate 决定
#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         mixed_precision=None, # 让 accelerate launch 的参数决定
#         kwargs_handlers=[
#             DistributedDataParallelKwargs(find_unused_parameters=True)
#         ]
#     )
#     device = accelerator.device
#     batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

#     # Log config
#     if accelerator.is_main_process:
#         if enable_mlflow:
#             try:
#                 mlflow.log_params({
#                     **click.get_current_context().params,
#                     'batch_size_total': batch_size_total,
#                 })
#             except:
#                 print('Failed to log config to MLFlow')
#         Path(workspace).mkdir(parents=True, exist_ok=True)
#         with Path(workspace).joinpath('config.json').open('w') as f:
#             json.dump(config, f, indent=4)

#     # Set seed
#     if seed is not None:
#         set_seed(seed, device_specific=True)

#     # Initialize model
#     print('Initialize model')
#     with accelerator.local_main_process_first():
#         from moge.model import import_model_class_by_version
#         MoGeModel = import_model_class_by_version(config['model_version'])      
#         model = MoGeModel(**config['model'])
#         # --- 临时添加这段代码来“查户口” ---
#         # print("================= 模型层名一览表 =================")
#         # for name, module in model.named_modules():
#         #     print(name)
#         # print("================================================")
#         # exit() # 打印完直接退出，不用跑训练

#     # ================= 🛡️ FSDP 策略 (保持原样) 🛡️ =================
#     if accelerator.distributed_type == accelerate.DistributedType.FSDP:
#         try:
#             if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
#                 block_module = model.encoder.backbone.blocks[0]
#                 block_class = type(block_module)
#                 print(f"\n[FSDP Auto-Config] ✅ Detected Transformer Layer Class: {block_class.__name__}")
                
#                 import functools
#                 from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                
#                 auto_wrap_policy = functools.partial(
#                     transformer_auto_wrap_policy,
#                     transformer_layer_cls={block_class},
#                 )
#                 accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
#                 print(f"[FSDP Auto-Config] 🚀 Policy successfully applied using PyTorch native FSDP!\n")
#             else:
#                 print("[FSDP Auto-Config] ⚠️ Warning: Could not locate backbone.blocks automatically.")
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             print(f"[FSDP Auto-Config] ❌ Critical Error: {e}")
#     # =======================================================================

#     count_total_parameters = sum(p.numel() for p in model.parameters())
#     print(f'🔥Total parameters: {count_total_parameters}')
#     count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f'🔥 Trainable parameters: {count_trainable} (Ratio: {count_trainable/count_total_parameters:.2%})')

#     # Set up EMA model
#     if enable_ema and accelerator.is_main_process:
#         ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
#         ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

#     # Set gradient checkpointing
#     if enable_gradient_checkpointing:
#         model.enable_gradient_checkpointing()
#     import warnings
#     warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
#     # Initalize optimizer & lr scheduler
#     optimizer = build_optimizer(model, config['optimizer'])
#     lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

#     count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
#     for i, count in enumerate(count_grouped_parameters):
#         print(f'- Group {i}: {count} parameters')

#     # Attempt to load checkpoint
#     checkpoint: Dict[str, Any]
#     with accelerator.local_main_process_first():
#         if checkpoint_path is None:
#             checkpoint = None
#         elif checkpoint_path.endswith('.pt'):
#             print(f'Load checkpoint: {checkpoint_path}')
#             checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
#         elif checkpoint_path == "latest": 
#             checkpoint_path = Path(workspace, 'checkpoint', 'latest.pt')
#             if checkpoint_path.exists():
#                 print(f'Load checkpoint: {checkpoint_path}')
#                 checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
#                 i_step = checkpoint['step']
#                 if 'model' not in checkpoint and (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
#                     print(f'Load model checkpoint: {checkpoint_model_path}')
#                     checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
#                 if 'optimizer' not in checkpoint and (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
#                     print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
#                     checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
#                 if enable_ema and accelerator.is_main_process:
#                     if 'ema_model' not in checkpoint and (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
#                         print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
#                         checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
#             else:
#                 print(f'No latest checkpoint found. Start from scratch.')
#                 checkpoint = None
#         else:
#             i_step = int(checkpoint_path)
#             checkpoint = {'step': i_step}
#             if (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
#                 print(f'Load model checkpoint: {checkpoint_model_path}')
#                 checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
#             if (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
#                 print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
#                 checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
#             if enable_ema and accelerator.is_main_process:
#                 if (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
#                     print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
#                     checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']

#     if checkpoint is None:
#         print('Initialize model weights')
#         with accelerator.local_main_process_first():
#             model.init_weights()
#         initial_step = 0
#     else:
#         model.load_state_dict(checkpoint['model'], strict=False)
#         if 'step' in checkpoint:
#             initial_step = checkpoint['step'] + 1
#         else:
#             initial_step = 0
#         if 'optimizer' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#         if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
#             ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
#         if 'lr_scheduler' in checkpoint:
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         del checkpoint
    
#     model, optimizer = accelerator.prepare(model, optimizer)
#     if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         from moge.model.utils import sync_ddp_hook
#         model.register_comm_hook(None, sync_ddp_hook)

#     with accelerator.local_main_process_first():
#         train_data_pipe = TrainDataLoaderPipeline(
#             config['data'], 
#             batch_size_forward,
#             num_load_workers=6, 
#             num_process_workers=6, 
#             buffer_size=32 
#         )

#     def _write_bytes_retry_loop(save_path: Path, data: bytes):
#         while True:
#             try:
#                 save_path.write_bytes(data)
#                 break
#             except Exception as e:
#                 print('Error while saving checkpoint, retrying in 1 minute: ', e)
#                 time.sleep(60)

#     # Ready to train
#     records = []
    
#     # 🟢 开启训练模式
#     model.train()

#     # ================= 🛡️ 护身符 1：全参数微调也必须冻结 BN 统计量 🛡️ =================
#     # 原因：你的 Batch Size 只有 8，如果让 Backbone 更新统计量，模型必崩。
#     # 方法：参数 (Weights) 保持 requires_grad=True，但模式设为 eval()
#     def freeze_bn_stats(m):
#         if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm, nn.SyncBatchNorm)):
#             m.eval() 
    
#     print("🛡️ [Safe Mode] Freezing BN statistics for full-finetuning...")
#     model.apply(freeze_bn_stats)
#     # ===========================================================================
    
#     with (
#         train_data_pipe,
#         tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
#         ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
#     ):  
#         # Get some batches for visualization
#         if accelerator.is_main_process:
#             batches_for_vis: List[Dict[str, torch.Tensor]] = []
#             num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
#             for _ in range(num_vis_images // batch_size_forward):
#                 batch = train_data_pipe.get()
#                 batches_for_vis.append(batch)

#         # Visualize GT
#         if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
#             save_dir = Path(workspace).joinpath('vis/gt')
#             for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
#                 image, gt_depth, gt_normal, gt_intrinsics, info = batch['image'], batch['depth'], batch['normal'], batch['intrinsics'], batch['info']
#                 gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
#                 for i_instance in range(batch['image'].shape[0]):
#                     idx = i_batch * batch_size_forward + i_instance
#                     image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#                     gt_depth_i = gt_depth[i_instance].numpy()
#                     gt_points_i = gt_points[i_instance].numpy()
#                     gt_normal_i = gt_normal[i_instance].numpy()
#                     save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
#                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
#                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(gt_points_i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
#                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(gt_depth_i), cv2.COLOR_RGB2BGR))
#                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal.png')), cv2.cvtColor(colorize_normal(gt_normal_i), cv2.COLOR_RGB2BGR))
#                     with save_dir.joinpath(f'{idx:04d}/info.json').open('w') as f:
#                         json.dump(info[i_instance], f)

#         if seed is not None:
#             set_seed(seed + initial_step, device_specific=True)   

#         # Training loop
#         for i_step in range(initial_step, num_iterations):

#             i_accumulate, weight_accumulate = 0, 0
#             while i_accumulate < gradient_accumulation_steps:
#                 batch = train_data_pipe.get()
#                 image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric']
#                 image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
#                 current_batch_size = image.shape[0]
                
#                 # NOTE: Skip all-invalid batches logic moved to later stage to prevent crash
                
#                 gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
#                 gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

#                 with accelerator.accumulate(model):
#                     if i_step <= config.get('low_resolution_training_steps', 0):
#                         num_tokens = config['model']['num_tokens_range'][0]
#                     else:
#                         num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
#                     # 🔥 [Fix 2: BF16 Support] 自动适配精度
#                     autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
#                     is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
#                     with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
#                         output = model(image, num_tokens=num_tokens)
                    
#                     pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])

#                     loss_list, weight_list = [], []
#                     for i in range(current_batch_size):
#                         # if label_type[i] == 'invalid':
#                         #     continue 
#                         current_label = str(label_type[i])
                        
#                         # 2. 直接检查 Config 里有没有这个 Key
#                         # 只要 Config['loss'] 里没定义怎么算这个 label 的 loss，我们就跳过
#                         if current_label not in config['loss']:
#                             # 仅在第一次遇到时打印，避免刷屏 (可选)
#                             # if accelerator.is_main_process:
#                             #     print(f"⚠️ [Skip] Label '{current_label}' not found in config.")
#                             continue
                            
#                         gt_metric_scale = None
#                         loss_dict, weight_dict, misc_dict = {}, {}, {}

#                         for k, v in config['loss'][label_type[i]].items():
#                             weight_dict[k] = v['weight']
#                             if v['function'] == 'affine_invariant_global_loss':
#                                 loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(pred_points[i], gt_points[i], **v['params'])
#                             elif v['function'] == 'affine_invariant_local_loss':
#                                 loss_dict[k], misc_dict[k] = affine_invariant_local_loss(
#                                         pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), 
#                                         gt_focal[i], gt_metric_scale, **v['params'])
#                             elif v['function'] == 'geometry_consistency_loss':
#                                 val, _ = geometry_consistency_loss(pred_points[i], gt_points[i])
#                             elif v['function'] == 'normal_loss':
#                                 loss_dict[k], misc_dict[k] = normal_loss(pred_points[i], gt_points[i])
#                             elif v['function'] == 'edge_loss':
#                                 loss_dict[k], misc_dict[k] = edge_loss(pred_points[i], gt_points[i])
#                             elif v['function'] == 'normal_map_loss':
#                                 loss_dict[k], misc_dict[k] = normal_map_loss(pred_normal[i], gt_normal[i])
#                             elif v['function'] == 'mask_bce_loss':
#                                 loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
#                             elif v['function'] == 'mask_l2_loss':
#                                 loss_dict[k], misc_dict[k] = mask_l2_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
#                             elif v['function'] == 'metric_scale_loss':
#                                 if is_metric[i] and pred_metric_scale is not None:
#                                     loss_dict[k], misc_dict[k] = metric_scale_loss(pred_metric_scale[i], gt_metric_scale)
#                             else:
#                                 raise ValueError(f'Undefined loss function: {v["function"]}')
                        
#                         weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
#                         loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                        
#                         loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
#                         loss_list.append(loss_)
                        
#                         if isinstance(loss_, torch.Tensor) and torch.isnan(loss_).item():
#                             pbar.write(f'NaN loss in process {accelerator.process_index}')

#                         safe_misc_dict = {}
#                         misc_dict = flatten_nested_dict(misc_dict)
#                         for k_misc, v_misc in misc_dict.items():
#                             k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
#                             if isinstance(v_misc, torch.Tensor):
#                                 safe_misc_dict[k_misc_str] = v_misc.detach().item() 
#                             else:
#                                 safe_misc_dict[k_misc_str] = v_misc

#                         safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
#                         records.append({**safe_loss_dict, **safe_misc_dict})

#                     # 🔥 [Fix 3: Ghost Loss] 处理全 Batch 无效的情况
#                     if len(loss_list) > 0:
#                         loss = sum(loss_list) / len(loss_list)
#                     else:
#                         if accelerator.is_main_process:
#                             pbar.write(f"⚠️ [Step {i_step}] Entire batch invalid! Using dummy loss.")
#                         # 构造关联梯度的 0 值，保持计算图连接
#                         loss = pred_points.sum() * 0.0 
                    
#                     accelerator.backward(loss)

#                     if accelerator.sync_gradients:
#                         # 🔥 [Fix 4: NaN 熔断] 防止梯度爆炸污染全参数微调的模型
#                         params_to_check = [p for p in model.parameters() if p.grad is not None]
#                         if len(params_to_check) > 0:
#                             grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
#                             if torch.isnan(grad_norm) or torch.isinf(grad_norm):
#                                 if accelerator.is_main_process:
#                                     pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping.")
#                                 optimizer.zero_grad()
#                                 continue
                        
#                         optimizer.step()
#                         optimizer.zero_grad()

#                 i_accumulate += 1

#             lr_scheduler.step()

#             if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
#                 if accelerator.distributed_type == accelerate.DistributedType.FSDP:
#                     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#                     with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
#                         ema_model.update_parameters(model)
#                 else:
#                     ema_model.update_parameters(model)

#             if i_step == initial_step or i_step % log_every == 0:
#                 records = [key_average(records)]
#                 records = accelerator.gather_for_metrics(records, use_gather_object=True)
#                 if accelerator.is_main_process:
#                     records = [
#                         {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in r.items()}
#                         for r in records
#                     ]
#                     records = key_average(records)
#                     if enable_mlflow:
#                         try:
#                             mlflow.log_metrics(records, step=i_step)
#                         except Exception as e:
#                             print(f'Error while logging metrics to mlflow: {e}')
#                 records = []

#             # Save checkponits ... (保持原样)
#             if accelerator.is_main_process and (i_step % save_every == 0):
#                 pbar.write(f'Save checkpoint: {i_step:08d}')
#                 Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)
                
#                 with io.BytesIO() as f:
#                     torch.save({
#                         'model_config': config['model'],
#                         'model': accelerator.unwrap_model(model).state_dict(),
#                     }, f)
#                     checkpoint_bytes = f.getvalue()
#                 save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), checkpoint_bytes)

#                 with io.BytesIO() as f:
#                     torch.save({
#                         'model_config': config['model'],
#                         'step': i_step,
#                         'optimizer': optimizer.state_dict(),
#                         'lr_scheduler': lr_scheduler.state_dict(),
#                     }, f)
#                     checkpoint_bytes = f.getvalue()
#                 save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'), checkpoint_bytes)
                
#                 if enable_ema:
#                     with io.BytesIO() as f:
#                         torch.save({
#                             'model_config': config['model'],
#                             'model': ema_model.module.state_dict(),
#                         }, f)
#                         checkpoint_bytes = f.getvalue()
#                     save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'), checkpoint_bytes)

#                 with io.BytesIO() as f:
#                     torch.save({
#                         'model_config': config['model'],
#                         'step': i_step,
#                     }, f)
#                     checkpoint_bytes = f.getvalue()
#                 save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes)
            
#             # Visualize ... (保持原样)
#             if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0):
#                 unwrapped_model = accelerator.unwrap_model(model)
#                 save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
#                 save_dir.mkdir(parents=True, exist_ok=True)
#                 with torch.inference_mode():
#                     for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
#                         image, gt_depth, gt_intrinsics = batch['image'], batch['depth'], batch['intrinsics']
#                         image, gt_depth, gt_intrinsics = image.to(device), gt_depth.to(device), gt_intrinsics.to(device)
#                         output = unwrapped_model.infer(image)
#                         pred_points = output['points'].cpu().numpy() if 'points' in output else None
#                         pred_depth = output['depth'].cpu().numpy() if 'depth' in output else None
#                         pred_mask =  output['mask'].cpu().numpy() if 'mask' in output else None
#                         pred_normal = output['normal'].cpu().numpy() if 'normal' in output else None
#                         image = (image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
#                         for i_instance in range(image.shape[0]):
#                             idx = i_batch * batch_size_forward + i_instance
#                             save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
#                             cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image[i_instance], cv2.COLOR_RGB2BGR))
#                             if pred_points is not None:
#                                 cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(pred_points[i_instance], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
#                             if pred_depth is not None:
#                                 cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(pred_depth[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
#                             if pred_normal is not None:
#                                 cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal_vis.png')), cv2.cvtColor(colorize_normal(pred_normal[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
            
#             pbar.set_postfix({'loss': loss.item()}, refresh=False)
#             pbar.update(1)

# if __name__ == '__main__':
#     main()