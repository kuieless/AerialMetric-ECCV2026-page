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
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size')
@click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
@click.option('--seed', type=int, default=0, help='Random seed')
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
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 自动适配混合精度
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True) 
        ]
    )
    device = accelerator.device
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

    # Log config
    if accelerator.is_main_process:
        if enable_mlflow:
            try:
                mlflow.log_params({
                    **click.get_current_context().params,
                    'batch_size_total': batch_size_total,
                })
            except:
                print('Failed to log config to MLFlow')
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
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
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={block_class},
                )
                accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
        except Exception as e:
            pass
    # =================================================

    # ================= 🟢 冻结 ViT Backbone & 启用 BN 保护 🟢 =================
    print("❄️ Configuring Freeze/Train modes...")
    
    # 1. 冻结 Backbone 参数
    frozen_count = 0
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
        for param in model.encoder.backbone.parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"✅ ViT Backbone frozen successfully! ({frozen_count} tensors frozen)")
        model.encoder.backbone.eval() 
    else:
        print("⚠️ Warning: Could not find 'model.encoder.backbone' to freeze.")

    # 2. 开启训练模式 (为了 Neck 和 Head)
    model.train()

    # 3. 强制冻结所有 BN 层的统计量 (Running Mean/Var)
    def freeze_bn_stats(m):
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm, nn.SyncBatchNorm)):
            m.eval() 

    print("🛡️ Applying BN Freeze Protection to the WHOLE model...")
    model.apply(freeze_bn_stats)
    # =================================================================================

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    # Initalize optimizer & lr scheduler
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
        initial_step = checkpoint.get('step', 0) + 1
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
        train_data_pipe = TrainDataLoaderPipeline(
            config['data'], 
            batch_size_forward,
            num_load_workers=6, 
            num_process_workers=6, 
            buffer_size=32 
        )

    def _write_bytes_retry_loop(save_path: Path, data: bytes):
        while True:
            try:
                save_path.write_bytes(data)
                break
            except Exception as e:
                print('Error while saving checkpoint, retrying in 1 minute: ', e)
                time.sleep(60)

    # Ready to train
    records = []
    
    # 🟢 再次确认 BN 状态
    model.apply(freeze_bn_stats)
    nan_log_file = Path(workspace).joinpath("nan_images_log.txt")

    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):  
        # Get some batches for visualization
        if accelerator.is_main_process:
            batches_for_vis: List[Dict[str, torch.Tensor]] = []
            num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
            for _ in range(num_vis_images // batch_size_forward):
                batch = train_data_pipe.get()
                batches_for_vis.append(batch)

        # if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
        #     # ... (可视化 GT 代码省略，保持不变) ...
        #     pass
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

        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)   
        
        # Training loop
        for i_step in range(initial_step, num_iterations):

            i_accumulate = 0
            
            # while i_accumulate < gradient_accumulation_steps:
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
                
                # 🛡️ 保护 1: 防止焦距计算除以 0
                fx = gt_intrinsics[..., 0, 0].float()
                fy = gt_intrinsics[..., 1, 1].float()
                gt_focal = (1.0 / torch.sqrt(1.0 / (fx ** 2 + 1e-6) + 1.0 / (fy ** 2 + 1e-6))).float()

                with accelerator.accumulate(model):
                    if i_step <= config.get('low_resolution_training_steps', 0):
                        num_tokens = config['model']['num_tokens_range'][0]
                    else:
                        num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
                    autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
                    is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
                    with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
                        output = model(image, num_tokens=num_tokens)
                    
                    pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])
                    
                    with torch.autocast(device_type=device.type, enabled=False):
                        valid_loss_list = []
                        
                        for i in range(current_batch_size):
                            current_label = str(label_type[i])
                            if current_label not in config['loss']: continue

                            p_points = pred_points[i].float()
                            p_normal = pred_normal[i].float() if pred_normal is not None else None
                            p_mask = pred_mask[i].float() if pred_mask is not None else None
                            p_metric_scale = pred_metric_scale[i].float() if pred_metric_scale is not None else None

                            g_points = gt_points[i].float()
                            g_normal = gt_normal[i].float()
                            g_mask_fin = gt_mask_fin[i]
                            g_mask_inf = gt_mask_inf[i]
                            g_focal = gt_focal[i]

                            gt_metric_scale = None 
                            loss_dict, weight_dict, misc_dict = {}, {}, {}
                            is_sample_broken = False 
                            
                            for k, v in config['loss'][current_label].items():
                                weight_dict[k] = v['weight']
                                term_loss = torch.tensor(0.0, device=device)

                                # 🛡️ 保护 2: 捕获由于 0方差 或 NaN 输入导致的函数崩溃
                                try:
                                    if v['function'] == 'affine_invariant_global_loss':
                                        term_loss, misc_dict[k], gt_metric_scale = affine_invariant_global_loss(p_points, g_points, **v['params'])
                                    elif v['function'] == 'affine_invariant_local_loss':
                                        term_loss, misc_dict[k] = affine_invariant_local_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), g_focal, gt_metric_scale, **v['params'])
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
                                    elif v['function'] == 'metric_scale_loss':
                                        if is_metric[i] and p_metric_scale is not None:
                                            term_loss, misc_dict[k] = metric_scale_loss(p_metric_scale, gt_metric_scale)
                                except Exception as e:
                                    bad_info = info[i]
                                    print(f"\n🚨 [Step {i_step}] 函数执行失败! 函数: {v['function']}, 数据集: {bad_info.get('dataset')}, 图片: {bad_info.get('filename')}, 错误: {str(e)}")
                                    is_sample_broken = True
                                    break
                                
                                loss_dict[k] = term_loss
                                
                                # 🛡️ 保护 3: 侦测单独一项的 NaN
                                if torch.isnan(term_loss).item() or torch.isinf(term_loss).item():
                                    bad_info = info[i]
                                    print(f"\n🚨 [Step {i_step}] 前向计算产生 NaN! 函数: {v['function']}, 数据集: {bad_info.get('dataset')}, 图片: {bad_info.get('filename')}")
                                    is_sample_broken = True
                                    break 
                            
                            # 如果该样本任意一项爆炸，放弃该样本
                            if is_sample_broken:
                                continue 

                            weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                            loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                            
                            sample_total_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                            
                            if not isinstance(sample_total_loss, torch.Tensor):
                                sample_total_loss = torch.tensor(sample_total_loss, dtype=torch.float32, device=device)

                            if torch.isnan(sample_total_loss).item() or torch.isinf(sample_total_loss).item():
                                continue
                            
                            # 🛡️ 保护 4: 数值硬截断，防止深度差异过大撑爆梯度
                            if sample_total_loss > 1000.0:
                                sample_total_loss = torch.clamp(sample_total_loss, max=1000.0)

                            valid_loss_list.append(sample_total_loss)
                            
                            safe_misc_dict = {}
                            misc_dict = flatten_nested_dict(misc_dict)
                            for k_misc, v_misc in misc_dict.items():
                                k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
                                if isinstance(v_misc, torch.Tensor):
                                    safe_misc_dict[k_misc_str] = v_misc.detach().float().item() 
                                else:
                                    safe_misc_dict[k_misc_str] = v_misc
                            safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                            records.append({**safe_loss_dict, **safe_misc_dict})

                        # 4. Batch 级聚合与 Backward
                        if len(valid_loss_list) > 0:
                            loss = sum(valid_loss_list) / len(valid_loss_list)
                            
                            # 🛡️ 保护 5: 终极侦测武器！用 detect_anomaly 包裹 backward
                            with torch.autograd.detect_anomaly():
                                accelerator.backward(loss)
                            
                            if accelerator.sync_gradients:
                                params_to_check = [p for p in model.parameters() if p.grad is not None]
                                if len(params_to_check) > 0:
                                    grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
                                    if torch.isnan(grad_norm).item() or torch.isinf(grad_norm).item():
                                        if accelerator.is_main_process:
                                            pbar.write(f"⚠️ [Step {i_step}] 梯度依然爆炸 (Norm={grad_norm.item()})! 跳过此步。请留意上方的🚨报错信息！")
                                        optimizer.zero_grad(set_to_none=True)
                                    else:
                                        optimizer.step()
                                        optimizer.zero_grad(set_to_none=True)
                                else:
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                            else:
                                pass 
                        else:
                            if accelerator.is_main_process:
                                pbar.write(f"⚠️ [Step {i_step}] 当前 Batch 的图片全包含非法数值！使用 Dummy Loss。")
                            dummy_loss = pred_points.sum() * 0.0
                            accelerator.backward(dummy_loss)
                            optimizer.zero_grad(set_to_none=True)

                i_accumulate += 1
                # try:
                #     batch = train_data_pipe.get()
                # except Exception as e:
                #     print(f"⚠️ Dataloader Error: {e}")
                #     continue

                # image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = \
                #     batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']
                
                # image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = \
                #     image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                
                # current_batch_size = image.shape[0]
                
                # gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                
                # # 🛡️ [防爆修改 1] 安全的 Focal 计算
                # # 防止 fx/fy 为 0 导致除以 0 产生 Inf
                # fx = gt_intrinsics[..., 0, 0].float()
                # fy = gt_intrinsics[..., 1, 1].float()
                # epsilon = 1e-6
                # gt_focal = (1.0 / torch.sqrt(1.0 / (fx ** 2 + epsilon) + 1.0 / (fy ** 2 + epsilon))).float()

                # with accelerator.accumulate(model):
                #     if i_step <= config.get('low_resolution_training_steps', 0):
                #         num_tokens = config['model']['num_tokens_range'][0]
                #     else:
                #         num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
                #     # 混合精度 Forward
                #     autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
                #     is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
                #     with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
                #         output = model(image, num_tokens=num_tokens)
                    
                #     pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])
                    
                #     # 🛡️ [防爆修改 2] 逐样本 Loss 清洗流程
                #     # 强制退出 FP16，使用 FP32 计算 Loss，避免溢出
                #     with torch.autocast(device_type=device.type, enabled=False):
                #         valid_loss_list = []  # 存活下来的 Loss
                        
                #         # 🔄 逐样本细粒度检查 (Per-sample Check)
                #         for i in range(current_batch_size):
                #             # 1. 基础数据准备
                #             current_label = str(label_type[i])
                #             if current_label not in config['loss']: continue

                #             # 转换为 FP32 防止溢出
                #             p_points = pred_points[i].float()
                #             p_normal = pred_normal[i].float() if pred_normal is not None else None
                #             p_mask = pred_mask[i].float() if pred_mask is not None else None
                #             p_metric_scale = pred_metric_scale[i].float() if pred_metric_scale is not None else None

                #             g_points = gt_points[i].float()
                #             g_normal = gt_normal[i].float()
                #             g_mask_fin = gt_mask_fin[i]
                #             g_mask_inf = gt_mask_inf[i]
                #             g_focal = gt_focal[i] # 已经是 float 了

                #             gt_metric_scale = None 
                #             loss_dict, weight_dict, misc_dict = {}, {}, {}
                #             is_sample_broken = False # 标记当前样本是否损坏
                            
                #             # 2. 计算各项 Loss
                #             for k, v in config['loss'][current_label].items():
                #                 weight_dict[k] = v['weight']
                                
                #                 term_loss = torch.tensor(0.0, device=device) # 临时变量

                #                 if v['function'] == 'affine_invariant_global_loss':
                #                     term_loss, misc_dict[k], gt_metric_scale = affine_invariant_global_loss(p_points, g_points, **v['params'])
                #                 elif v['function'] == 'affine_invariant_local_loss':
                #                     term_loss, misc_dict[k] = affine_invariant_local_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), g_focal, gt_metric_scale, **v['params'])
                #                 elif v['function'] == 'geometry_consistency_loss':
                #                     val, _ = geometry_consistency_loss(p_points, g_points)
                #                     term_loss = val # 假设这里需要反向传播
                #                 elif v['function'] == 'normal_loss':
                #                     term_loss, misc_dict[k] = normal_loss(p_points, g_points)
                #                 elif v['function'] == 'edge_loss':
                #                     term_loss, misc_dict[k] = edge_loss(p_points, g_points)
                #                 elif v['function'] == 'normal_map_loss':
                #                     term_loss, misc_dict[k] = normal_map_loss(p_normal, g_normal)
                #                 elif v['function'] == 'mask_bce_loss':
                #                     term_loss, misc_dict[k] = mask_bce_loss(p_mask, g_mask_fin, g_mask_inf)
                #                 elif v['function'] == 'mask_l2_loss':
                #                     term_loss, misc_dict[k] = mask_l2_loss(p_mask, g_mask_fin, g_mask_inf)
                #                 elif v['function'] == 'multi_scale_gradient_loss':
                #                     term_loss = multi_scale_gradient_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), params=v.get('params', {}))
                #                     misc_dict[k] = term_loss
                #                 elif v['function'] == 'metric_scale_loss':
                #                     if is_metric[i] and p_metric_scale is not None:
                #                         term_loss, misc_dict[k] = metric_scale_loss(p_metric_scale, gt_metric_scale)
                                
                #                 loss_dict[k] = term_loss
                                
                #                 # 🔥 立即检查单个 Loss 项是否 NaN/Inf
                #                 if torch.isnan(term_loss) or torch.isinf(term_loss):
                #                     is_sample_broken = True
                #                     # 不需要 break，继续算完其他项只是浪费一点点计算量，但逻辑更简单
                            
                #             # 如果单个 Loss 组件坏了，直接跳过当前样本
                #             if is_sample_broken:
                #                 bad_info = info[i]
                #                 if accelerator.is_main_process:
                #                     print(f"⚠️ [Step {i_step}] Skip Bad Image (NaN Component): {bad_info.get('dataset')}/{bad_info.get('filename')}")
                #                 continue 

                #             # 3. 聚合当前样本的总 Loss
                #             # weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                #             # loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                            
                #             # sample_total_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
                            
                #             # # 🛡️ [防爆修改 3] Loss 最终安检
                #             # if torch.isnan(sample_total_loss) or torch.isinf(sample_total_loss):


                #             # 3. 聚合当前样本的总 Loss
                #             weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                #             loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}

                #             # 强转 start 为 Tensor，保证 sum 的结果绝对是 Tensor
                #             sample_total_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))

                #             # 如果因为特殊原因它依然是个 float，再做一次兜底转换
                #             if not isinstance(sample_total_loss, torch.Tensor):
                #                 sample_total_loss = torch.tensor(sample_total_loss, dtype=torch.float32, device=device)

                #             # 🛡️ [防爆修改 3] Loss 最终安检 (加上 .item() 更安全)
                #             if torch.isnan(sample_total_loss).item() or torch.isinf(sample_total_loss).item():
                #                 bad_info = info[i]
                #                 if accelerator.is_main_process:
                #                     print(f"⚠️ [Step {i_step}] Skip Bad Image (Total NaN): {bad_info.get('dataset')}/{bad_info.get('filename')}")
                #                 continue
                            
                #             # 🛡️ [防爆修改 4] 数值截断 (Clamping)
                #             # 如果 Loss 巨大但不是 NaN (例如 10000.0)，截断到 100.0，防止梯度爆炸
                #             if sample_total_loss > 1000.0:
                #                 if accelerator.is_main_process:
                #                     print(f"⚠️ [Step {i_step}] Loss Spike ({sample_total_loss.item():.2f}) -> Clamped to 100.0")
                #                 sample_total_loss = torch.clamp(sample_total_loss, max=100.0)

                #             # ✅ 通过检查，加入有效列表
                #             valid_loss_list.append(sample_total_loss)
                            
                #             # 记录 Log
                #             safe_misc_dict = {}
                #             misc_dict = flatten_nested_dict(misc_dict)
                #             for k_misc, v_misc in misc_dict.items():
                #                 k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
                #                 if isinstance(v_misc, torch.Tensor):
                #                     safe_misc_dict[k_misc_str] = v_misc.detach().float().item() 
                #                 else:
                #                     safe_misc_dict[k_misc_str] = v_misc
                #             safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                #             records.append({**safe_loss_dict, **safe_misc_dict})

                #         # 4. Batch 级聚合与 Backward
                #         if len(valid_loss_list) > 0:
                #             # 只对有效的样本求平均
                #             loss = sum(valid_loss_list) / len(valid_loss_list)
                            
                #             accelerator.backward(loss)
                            
                #             # 5. 梯度裁剪 (带 NaN 检查)
                #             if accelerator.sync_gradients:
                #                 params_to_check = [p for p in model.parameters() if p.grad is not None]
                #                 if len(params_to_check) > 0:
                #                     grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
                #                     if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                #                         if accelerator.is_main_process:
                #                             pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping update.")
                #                         optimizer.zero_grad(set_to_none=True)
                #                     else:
                #                         optimizer.step()
                #                         optimizer.zero_grad(set_to_none=True)
                #                 else:
                #                     optimizer.step()
                #                     optimizer.zero_grad(set_to_none=True)
                #             else:
                #                 # 非 Sync 步 (Accumulation 阶段)
                #                 pass 
                            
                #         else:
                #             # 整个 Batch 全军覆没 (全是坏图)
                #             if accelerator.is_main_process:
                #                 pbar.write(f"⚠️ [Step {i_step}] All images in batch were invalid! Using dummy loss.")
                            
                #             # 构造 Dummy Loss 维持 DDP 及其同步
                #             # 必须让所有进程都执行 backward，否则 DDP 会 hang 住
                #             dummy_loss = pred_points.sum() * 0.0
                #             accelerator.backward(dummy_loss)
                #             optimizer.zero_grad(set_to_none=True)

                # i_accumulate += 1

            lr_scheduler.step()
            
            # EMA Update
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
                    records = [
                        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in r.items()}
                        for r in records
                    ]
                    records = key_average(records)
                    if enable_mlflow:
                        try:
                            mlflow.log_metrics(records, step=i_step)
                        except Exception as e:
                            print(f'Error while logging metrics to mlflow: {e}')
                records = []

            # Save checkponits
            if accelerator.is_main_process and (i_step % save_every == 0):
                pbar.write(f'Save checkpoint: {i_step:08d}')
                Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)
                
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'model': accelerator.unwrap_model(model).state_dict(),
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), checkpoint_bytes)

                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'step': i_step,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'), checkpoint_bytes)
                
                if enable_ema:
                    with io.BytesIO() as f:
                        torch.save({
                            'model_config': config['model'],
                            'model': ema_model.module.state_dict(),
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'), checkpoint_bytes)

                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'step': i_step,
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes)
            
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


# # import os
# # from pathlib import Path
# # import sys
# # # 保持原有的路径添加逻辑
# # if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
# #     sys.path.insert(0, _package_root)
# # import json
# # import time
# # import random
# # from typing import *
# # import itertools
# # from contextlib import nullcontext
# # from concurrent.futures import ThreadPoolExecutor
# # import io

# # import numpy as np
# # import cv2
# # from PIL import Image
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.version
# # import accelerate
# # from accelerate import Accelerator, DistributedDataParallelKwargs
# # from accelerate.utils import set_seed
# # import utils3d
# # import click
# # from tqdm import tqdm, trange
# # import mlflow
# # torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

# # from moge.train.dataloader import TrainDataLoaderPipeline
# # from moge.train.losses import (
# #     affine_invariant_global_loss,
# #     affine_invariant_local_loss, 
# #     edge_loss,
# #     normal_loss, 
# #     mask_l2_loss, 
# #     mask_bce_loss,
# #     metric_scale_loss,
# #     normal_map_loss,
# #     monitoring, 

# # )
# # from moge.train.utils import build_optimizer, build_lr_scheduler
# # from moge.utils.geometry_torch import intrinsics_to_fov
# # from moge.utils.vis import colorize_depth, colorize_normal
# # from moge.utils.tools import key_average, recursive_replace, CallbackOnException, flatten_nested_dict
# # from moge.test.metrics import compute_metrics


# # def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
# #     if params is None: params = {}
# #     scales = params.get('scales', 4)
# #     total_loss = 0
# #     valid_scales = 0

# #     for scale in range(scales):
# #         step = 2 ** scale
# #         if pred.shape[2] <= step or pred.shape[3] <= step: break

# #         # Calculate gradients
# #         pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
# #         gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
# #         pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
# #         gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

# #         # Create a validity mask (1 where data is good, 0 where data is NaN/Inf)
# #         # We must check both GT and Pred for validity
# #         valid_mask_x = torch.isfinite(gt_grad_x) & torch.isfinite(pred_grad_x)
# #         valid_mask_y = torch.isfinite(gt_grad_y) & torch.isfinite(pred_grad_y)

# #         # Combine with user-provided mask if it exists
# #         if mask is not None:
# #             mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
# #             mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
# #             valid_mask_x = valid_mask_x & mask_x
# #             valid_mask_y = valid_mask_y & mask_y

# #         # Compute Loss only on valid pixels
# #         if valid_mask_x.sum() > 0:
# #             diff_x = torch.abs(pred_grad_x - gt_grad_x)
# #             # Only sum errors where mask is True
# #             loss_x = diff_x[valid_mask_x].mean()
# #         else:
# #             loss_x = 0.0

# #         if valid_mask_y.sum() > 0:
# #             diff_y = torch.abs(pred_grad_y - gt_grad_y)
# #             loss_y = diff_y[valid_mask_y].mean()
# #         else:
# #             loss_y = 0.0

# #         total_loss += (loss_x + loss_y)
# #         valid_scales += 1

# #     if valid_scales > 0:
# #         return total_loss / valid_scales
# #     return torch.tensor(0.0, device=pred.device, requires_grad=True)

# # @click.command()
# # @click.option('--config', 'config_path', type=str, default='configs/debug.json')
# # @click.option('--workspace', type=str, default='workspace/debug', help='Path to the workspace')
# # @click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load. "latest" to load latest checkpoint in workspace, integer to load by step number')
# # @click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
# # @click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
# # @click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
# # @click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
# # @click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
# # @click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
# # @click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
# # @click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
# # @click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
# # @click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size')
# # @click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
# # @click.option('--seed', type=int, default=0, help='Random seed')
# # def main(
# #     config_path: str,
# #     workspace: str,
# #     checkpoint_path: str,
# #     batch_size_forward: int,
# #     gradient_accumulation_steps: int,
# #     enable_gradient_checkpointing: bool,
# #     enable_mixed_precision: bool,
# #     enable_ema: bool,
# #     num_iterations: int,
# #     save_every: int,
# #     log_every: int,
# #     vis_every: int,
# #     num_vis_images: int,
# #     enable_mlflow: bool,
# #     seed: Optional[int],
# # ):
# #     # Load config
# #     with open(config_path, 'r') as f:
# #         config = json.load(f)
    
# #     # 🔥 [Fix 1] 自动适配混合精度
# #     accelerator = Accelerator(
# #         gradient_accumulation_steps=gradient_accumulation_steps,
# #         mixed_precision=None,
# #         kwargs_handlers=[
# #             DistributedDataParallelKwargs(find_unused_parameters=True) 
# #         ]
# #     )
# #     device = accelerator.device
# #     batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

# #     # Log config
# #     if accelerator.is_main_process:
# #         if enable_mlflow:
# #             try:
# #                 mlflow.log_params({
# #                     **click.get_current_context().params,
# #                     'batch_size_total': batch_size_total,
# #                 })
# #             except:
# #                 print('Failed to log config to MLFlow')
# #         Path(workspace).mkdir(parents=True, exist_ok=True)
# #         with Path(workspace).joinpath('config.json').open('w') as f:
# #             json.dump(config, f, indent=4)

# #     # Set seed
# #     if seed is not None:
# #         set_seed(seed, device_specific=True)

# #     # Initialize model
# #     print('Initialize model')
# #     with accelerator.local_main_process_first():
# #         from moge.model import import_model_class_by_version
# #         MoGeModel = import_model_class_by_version(config['model_version'])      
# #         model = MoGeModel(**config['model'])

# #     # ================= 🛡️ FSDP 策略 (保持原样) 🛡️ =================
# #     if accelerator.distributed_type == accelerate.DistributedType.FSDP:
# #         try:
# #             if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
# #                 block_module = model.encoder.backbone.blocks[0]
# #                 block_class = type(block_module)
# #                 print(f"\n[FSDP Auto-Config] ✅ Detected Transformer Layer Class: {block_class.__name__}")
                
# #                 import functools
# #                 from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                
# #                 auto_wrap_policy = functools.partial(
# #                     transformer_auto_wrap_policy,
# #                     transformer_layer_cls={block_class},
# #                 )
# #                 accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
# #                 print(f"[FSDP Auto-Config] 🚀 Policy successfully applied using PyTorch native FSDP!\n")
# #             else:
# #                 print("[FSDP Auto-Config] ⚠️ Warning: Could not locate backbone.blocks automatically.")
# #         except Exception as e:
# #             import traceback
# #             traceback.print_exc()
# #             print(f"[FSDP Auto-Config] ❌ Critical Error: {e}")
# #     # =======================================================================

# #     # ================= 🟢 [关键修复] 冻结 ViT Backbone & 启用 BN 保护 🟢 =================
# #     print("❄️ Configuring Freeze/Train modes...")
    
# #     # 1. 冻结 Backbone 参数 (Weights)
# #     frozen_count = 0
# #     if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
# #         for param in model.encoder.backbone.parameters():
# #             param.requires_grad = False
# #             frozen_count += 1
# #         print(f"✅ ViT Backbone frozen successfully! ({frozen_count} tensors frozen)")
# #         model.encoder.backbone.eval() 
# #     else:
# #         print("⚠️ Warning: Could not find 'model.encoder.backbone' to freeze.")

# #     # 2. 开启训练模式 (为了 Neck 和 Head)
# #     model.train()

# #     # 3. [护身符] 强制冻结所有 BN 层的统计量 (Running Mean/Var)
# #     # 即使是微调 Neck，BatchSize=8 时也绝对不能动统计量！
# #     def freeze_bn_stats(m):
# #         if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm, nn.SyncBatchNorm)):
# #             m.eval() 

# #     print("🛡️ Applying BN Freeze Protection to the WHOLE model (including Neck/Head)...")
# #     model.apply(freeze_bn_stats)
# #     # =================================================================================

# #     count_total_parameters = sum(p.numel() for p in model.parameters())
# #     print(f'🔥Total parameters: {count_total_parameters}')
# #     count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f'🔥 Trainable parameters: {count_trainable} (Ratio: {count_trainable/count_total_parameters:.2%})')

# #     # Set up EMA model
# #     if enable_ema and accelerator.is_main_process:
# #         ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
# #         ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

# #     # Set gradient checkpointing
# #     if enable_gradient_checkpointing:
# #         model.enable_gradient_checkpointing()
# #     import warnings
# #     warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
# #     # Initalize optimizer & lr scheduler
# #     optimizer = build_optimizer(model, config['optimizer'])
# #     lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

# #     count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
# #     for i, count in enumerate(count_grouped_parameters):
# #         print(f'- Group {i}: {count} parameters')

# #     # Attempt to load checkpoint (...保持原样...)
# #     checkpoint: Dict[str, Any]
# #     with accelerator.local_main_process_first():
# #         if checkpoint_path is None:
# #             checkpoint = None
# #         elif checkpoint_path.endswith('.pt'):
# #             print(f'Load checkpoint: {checkpoint_path}')
# #             checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
# #         elif checkpoint_path == "latest": 
# #             checkpoint_path = Path(workspace, 'checkpoint', 'latest.pt')
# #             if checkpoint_path.exists():
# #                 print(f'Load checkpoint: {checkpoint_path}')
# #                 checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
# #                 i_step = checkpoint['step']
# #                 if 'model' not in checkpoint and (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
# #                     print(f'Load model checkpoint: {checkpoint_model_path}')
# #                     checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
# #                 if 'optimizer' not in checkpoint and (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
# #                     print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
# #                     checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
# #                 if enable_ema and accelerator.is_main_process:
# #                     if 'ema_model' not in checkpoint and (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
# #                         print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
# #                         checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
# #             else:
# #                 print(f'No latest checkpoint found. Start from scratch.')
# #                 checkpoint = None
# #         else:
# #             i_step = int(checkpoint_path)
# #             checkpoint = {'step': i_step}
# #             if (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
# #                 print(f'Load model checkpoint: {checkpoint_model_path}')
# #                 checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
# #             if (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
# #                 print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
# #                 checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
# #             if enable_ema and accelerator.is_main_process:
# #                 if (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
# #                     print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
# #                     checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']

# #     if checkpoint is None:
# #         print('Initialize model weights')
# #         with accelerator.local_main_process_first():
# #             model.init_weights()
# #         initial_step = 0
# #     else:
# #         model.load_state_dict(checkpoint['model'], strict=False)
# #         if 'step' in checkpoint:
# #             initial_step = checkpoint['step'] + 1
# #         else:
# #             initial_step = 0
# #         if 'optimizer' in checkpoint:
# #             optimizer.load_state_dict(checkpoint['optimizer'])
# #         if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
# #             ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
# #         if 'lr_scheduler' in checkpoint:
# #             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
# #         del checkpoint
    
# #     model, optimizer = accelerator.prepare(model, optimizer)
# #     if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
# #         from moge.model.utils import sync_ddp_hook
# #         model.register_comm_hook(None, sync_ddp_hook)

# #     with accelerator.local_main_process_first():
# #         train_data_pipe = TrainDataLoaderPipeline(
# #             config['data'], 
# #             batch_size_forward,
# #             num_load_workers=6, 
# #             num_process_workers=6, 
# #             buffer_size=32 
# #         )

# #     def _write_bytes_retry_loop(save_path: Path, data: bytes):
# #         while True:
# #             try:
# #                 save_path.write_bytes(data)
# #                 break
# #             except Exception as e:
# #                 print('Error while saving checkpoint, retrying in 1 minute: ', e)
# #                 time.sleep(60)

# #     # Ready to train
# #     records = []
    
# #     # 🟢 再次确认 BN 状态 (防止 Accelerator prepare 之后被重置)
# #     model.apply(freeze_bn_stats)
    
# #     with (
# #         train_data_pipe,
# #         tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
# #         ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
# #     ):  
# #         # Get some batches for visualization (...保持原样...)
# #         if accelerator.is_main_process:
# #             batches_for_vis: List[Dict[str, torch.Tensor]] = []
# #             num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
# #             for _ in range(num_vis_images // batch_size_forward):
# #                 batch = train_data_pipe.get()
# #                 batches_for_vis.append(batch)

# #         if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
# #             save_dir = Path(workspace).joinpath('vis/gt')
# #             for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
# #                 image, gt_depth, gt_normal, gt_intrinsics, info = batch['image'], batch['depth'], batch['normal'], batch['intrinsics'], batch['info']
# #                 gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
# #                 for i_instance in range(batch['image'].shape[0]):
# #                     idx = i_batch * batch_size_forward + i_instance
# #                     image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
# #                     gt_depth_i = gt_depth[i_instance].numpy()
# #                     gt_points_i = gt_points[i_instance].numpy()
# #                     gt_normal_i = gt_normal[i_instance].numpy()
# #                     save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
# #                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
# #                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(gt_points_i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
# #                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(gt_depth_i), cv2.COLOR_RGB2BGR))
# #                     cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal.png')), cv2.cvtColor(colorize_normal(gt_normal_i), cv2.COLOR_RGB2BGR))
# #                     with save_dir.joinpath(f'{idx:04d}/info.json').open('w') as f:
# #                         json.dump(info[i_instance], f)

# #         if seed is not None:
# #             set_seed(seed + initial_step, device_specific=True)   
# #         nan_log_file = Path(workspace).joinpath("nan_images_log.txt")
# #         # Training loop
# #         for i_step in range(initial_step, num_iterations):

# #             i_accumulate = 0
            
# #             # [优化] 删除了 check_model_weights 循环，保持极速

# #             while i_accumulate < gradient_accumulation_steps:
# #                 try:
# #                     batch = train_data_pipe.get()
# #                 except Exception as e:
# #                     print(f"⚠️ Dataloader Error: {e}")
# #                     continue

# #                 image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = \
# #                     batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']
                
# #                 image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = \
# #                     image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                
# #                 current_batch_size = image.shape[0]
                
# #                 gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
# #                 # ⚠️ 确保 focal 是 float
# #                 gt_focal = (1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5).float()

# #                 with accelerator.accumulate(model):
# #                     # ... Token 逻辑 ...
# #                     if i_step <= config.get('low_resolution_training_steps', 0):
# #                         num_tokens = config['model']['num_tokens_range'][0]
# #                     else:
# #                         num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
# #                     # 混合精度 Forward
# #                     autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
# #                     is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
# #                     with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
# #                         output = model(image, num_tokens=num_tokens)
                    
# #                     pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])
                    
# #                     # 🔥 强制 FP32 计算 Loss
# #                     # with torch.autocast(device_type=device.type, enabled=False):
# #                     #     loss_list = [] # 移除了 weight_list，因为我们在下面直接求和了
                        
# #                     #     # 🔄 逐样本计算 Loss (Per-sample Loss Calculation)
# #                     #     for i in range(current_batch_size):
# #                     #         current_label = str(label_type[i])
# #                     #         if current_label not in config['loss']: continue
                            
# #                     #         # 1. 强制 FP32 转换 (数值稳定性关键)
# #                     #         p_points = pred_points[i].float()
# #                     #         p_normal = pred_normal[i].float() if pred_normal is not None else None
# #                     #         p_mask = pred_mask[i].float() if pred_mask is not None else None
# #                     #         p_metric_scale = pred_metric_scale[i].float() if pred_metric_scale is not None else None
                            
# #                     #         g_points = gt_points[i].float()
# #                     #         g_normal = gt_normal[i].float()
# #                     #         # ✅ Mask 保持 Bool 类型 (修复之前的报错)
# #                     #         g_mask_fin = gt_mask_fin[i]
# #                     #         g_mask_inf = gt_mask_inf[i]
# #                     #         g_focal = gt_focal[i] 

# #                     #         gt_metric_scale = None 
# #                     #         loss_dict, weight_dict, misc_dict = {}, {}, {}
# #                     with torch.autocast(device_type=device.type, enabled=False):
# #                         valid_loss_list = []  # 存活下来的 Loss
                        
# #                         # 🔄 逐样本细粒度检查 (Per-sample Check)
# #                         for i in range(current_batch_size):
# #                             # 1. 基础数据准备
# #                             current_label = str(label_type[i])
# #                             if current_label not in config['loss']: continue

# #                             # 转换为 FP32 防止溢出
# #                             p_points = pred_points[i].float()
# #                             g_points = gt_points[i].float()
                            
# #                             # 2. 单样本 Loss 计算
# #                             loss_dict, weight_dict = {}, {}
# #                             is_sample_broken = False # 标记当前样本是否损坏
                

# #                             # 2. 计算各项 Loss
# #                             for k, v in config['loss'][current_label].items():
# #                                 weight_dict[k] = v['weight']
                                
# #                                 if v['function'] == 'affine_invariant_global_loss':
# #                                     loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(p_points, g_points, **v['params'])
# #                                 elif v['function'] == 'affine_invariant_local_loss':
# #                                     loss_dict[k], misc_dict[k] = affine_invariant_local_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), g_focal, gt_metric_scale, **v['params'])
# #                                 elif v['function'] == 'geometry_consistency_loss':
# #                                     val, _ = geometry_consistency_loss(p_points, g_points)
# #                                 elif v['function'] == 'normal_loss':
# #                                     loss_dict[k], misc_dict[k] = normal_loss(p_points, g_points)
# #                                 elif v['function'] == 'edge_loss':
# #                                     loss_dict[k], misc_dict[k] = edge_loss(p_points, g_points)
# #                                 elif v['function'] == 'normal_map_loss':
# #                                     loss_dict[k], misc_dict[k] = normal_map_loss(p_normal, g_normal)
# #                                 elif v['function'] == 'mask_bce_loss':
# #                                     loss_dict[k], misc_dict[k] = mask_bce_loss(p_mask, g_mask_fin, g_mask_inf)
# #                                 elif v['function'] == 'mask_l2_loss':
# #                                     loss_dict[k], misc_dict[k] = mask_l2_loss(p_mask, g_mask_fin, g_mask_inf)
# #                                 elif v['function'] == 'multi_scale_gradient_loss':
# #                                     loss_dict[k] = multi_scale_gradient_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), params=v.get('params', {}))
# #                                     misc_dict[k] = loss_dict[k]
# #                                 elif v['function'] == 'metric_scale_loss':
# #                                     if is_metric[i] and p_metric_scale is not None:
# #                                         loss_dict[k], misc_dict[k] = metric_scale_loss(p_metric_scale, gt_metric_scale)
                            
# #                             # 3. 聚合当前样本的 Loss
# #                             weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
# #                             loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                            
# #                             sample_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
                            
# #                             # 🕵️‍♀️ 核心功能：检测单张图片的 Loss 是否异常 (NaN/Inf)
# #                             if isinstance(sample_loss, torch.Tensor) and (torch.isnan(sample_loss) or torch.isinf(sample_loss)):
# #                                 # 获取坏图信息
# #                                 bad_info = info[i]
# #                                 bad_msg = f"[Step {i_step}] ⚠️ Skip Bad Image: {bad_info.get('dataset', '?')}/{bad_info.get('filename', '?')} (Loss={sample_loss.item()})"
                                
# #                                 # 打印到控制台
# #                                 if accelerator.is_main_process:
# #                                     print(bad_msg)
# #                                     # 写入日志文件
# #                                     with open(nan_log_file, "a") as f:
# #                                         f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {bad_msg}\n")
                                
# #                                 # ❌ 跳过：不把这个 Loss 加入 loss_list
# #                                 continue 
                            
# #                             # ✅ 正常：加入列表
# #                             loss_list.append(sample_loss)
                            
# #                             # 记录 Log (仅记录正常的)
# #                             safe_misc_dict = {}
# #                             misc_dict = flatten_nested_dict(misc_dict)
# #                             for k_misc, v_misc in misc_dict.items():
# #                                 k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
# #                                 if isinstance(v_misc, torch.Tensor):
# #                                     safe_misc_dict[k_misc_str] = v_misc.detach().float().item() 
# #                                 else:
# #                                     safe_misc_dict[k_misc_str] = v_misc
# #                             safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
# #                             records.append({**safe_loss_dict, **safe_misc_dict})

# #                     # 4. 计算 Batch 总 Loss
# #                     if len(loss_list) > 0:
# #                         # 正常取平均
# #                         loss = sum(loss_list) / len(loss_list)
# #                     else:
# #                         # 如果这一个 Batch 所有图全是坏的 (Loss 都是 NaN)
# #                         if accelerator.is_main_process:
# #                             pbar.write(f"⚠️ [Step {i_step}] All images in batch are invalid! Using dummy loss.")
                        
# #                         # 使用 Dummy Loss 维持计算图，但不产生梯度
# #                         # 只要 pred_points 参与计算，check_unused_parameters 就不会报错
# #                         loss = pred_points.sum() * 0.0 
                    
# #                     # 5. Backward 前的最后防线 (防止 Dummy Loss 也是 NaN)
# #                     if torch.isnan(loss) or torch.isinf(loss):
# #                         print(f"⚠️ [Step {i_step}] Final Loss is NaN/Inf! Skipping backward.")
# #                         optimizer.zero_grad(set_to_none=True)
# #                         i_accumulate += 1 
# #                         continue 

# #                     accelerator.backward(loss)

# #                     # 6. 梯度裁剪与 Step
# #                     if accelerator.sync_gradients:
# #                         params_to_check = [p for p in model.parameters() if p.grad is not None]
# #                         if len(params_to_check) > 0:
# #                             grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
# #                             if torch.isnan(grad_norm) or torch.isinf(grad_norm):
# #                                 if accelerator.is_main_process:
# #                                     pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping.")
# #                                 optimizer.zero_grad(set_to_none=True)
# #                             else:
# #                                 optimizer.step()
# #                                 optimizer.zero_grad(set_to_none=True)
# #                         else:
# #                             optimizer.step()
# #                             optimizer.zero_grad(set_to_none=True)

# #                 i_accumulate += 1

# #             lr_scheduler.step()
            
# #             # EMA Update
# #             if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
# #                 if accelerator.distributed_type == accelerate.DistributedType.FSDP:
# #                     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# #                     with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
# #                         ema_model.update_parameters(model)
# #                 else:
# #                     ema_model.update_parameters(model)
# #         # for i_step in range(initial_step, num_iterations):

# #         #     i_accumulate, weight_accumulate = 0, 0

# #         #     if i_step % 10 == 0: # 不需要每步都查，每10步查一次
# #         #         has_nan_params = False
# #         #         for name, param in model.named_parameters():
# #         #             if torch.isnan(param).any() or torch.isinf(param).any():
# #         #                 print(f"💀 CRITICAL: Found NaN/Inf in model weights at step {i_step}!")
# #         #                 print(f" -> Layer: {name}")
# #         #                 print(f" -> Max: {param.max()}, Min: {param.min()}")
# #         #                 has_nan_params = True
# #         #                 break
# #         #         if has_nan_params:
# #         #             raise ValueError("Model weights have become NaN. Stop training.")
# #         #     while i_accumulate < gradient_accumulation_steps:
# #         #         batch = train_data_pipe.get()
# #         #         image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric']
# #         #         image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
# #         #         current_batch_size = image.shape[0]
                
# #         #         # NOTE: 移除了这里的 continue 跳过逻辑，改用下面的 Ghost Loss

# #         #         gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
# #         #         gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

# #         #         with accelerator.accumulate(model):
# #         #             if i_step <= config.get('low_resolution_training_steps', 0):
# #         #                 num_tokens = config['model']['num_tokens_range'][0]
# #         #             else:
# #         #                 num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
# #         #             # 🔥 [Fix 2] 自动混合精度
# #         #             autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
# #         #             is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
# #         #             with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
# #         #                 output = model(image, num_tokens=num_tokens)
                    
# #         #             pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])
# #         #             with torch.autocast(device_type=device.type, enabled=False):
# #         #                 loss_list, weight_list = [], []
                        
# #         #                 for i in range(current_batch_size):
# #         #                     # ================= 🛡️ Key 检查 =================
# #         #                     current_label = str(label_type[i])
# #         #                     if current_label not in config['loss']:
# #         #                         continue
                            
# #         #                     # ================= 🛡️ [Fix 2] 强制类型转换 (BF16 -> FP32) =================
# #         #                     # 即使外面关了 autocast，Tensor 本身可能还是 BF16，必须手动转 float()
# #         #                     # 1. 转换 Prediction
# #         #                     p_points = pred_points[i].float()
# #         #                     p_normal = pred_normal[i].float() if pred_normal is not None else None
# #         #                     p_mask = pred_mask[i].float() if pred_mask is not None else None
# #         #                     p_metric_scale = pred_metric_scale[i].float() if pred_metric_scale is not None else None
                            
# #         #                     # 2. 转换 Ground Truth
# #         #                     g_points = gt_points[i].float()
# #         #                     g_normal = gt_normal[i].float()
# #         #                     g_mask_fin = gt_mask_fin[i].float()
# #         #                     g_mask_inf = gt_mask_inf[i].float()
# #         #                     g_focal = gt_focal[i].float()   # ⚠️ 这个做除数非常关键，必须 FP32
                            
# #         #                     # 初始化变量
# #         #                     gt_metric_scale = None 
# #         #                     loss_dict, weight_dict, misc_dict = {}, {}, {}

# #         #                     # ================= 🛡️ Loss 计算 (使用 p_ 和 g_ 开头的变量) =================
# #         #                     for k, v in config['loss'][current_label].items():
# #         #                         weight_dict[k] = v['weight']
                                
# #         #                         if v['function'] == 'affine_invariant_global_loss':
# #         #                             # 注意返回值接收：loss, misc, scale
# #         #                             loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(
# #         #                                 p_points, g_points, **v['params']
# #         #                             )
                                    
# #         #                         elif v['function'] == 'affine_invariant_local_loss':
# #         #                             # 注意：输入需要 unsqueeze 变回 (1, C, H, W) 形状以适配函数接口
# #         #                             # 此时 gt_metric_scale 已经是 FP32 了（因为它是上面 global_loss 的输出）
# #         #                             loss_dict[k], misc_dict[k] = affine_invariant_local_loss(
# #         #                                 p_points.unsqueeze(0), g_points.unsqueeze(0), 
# #         #                                 g_focal, gt_metric_scale, **v['params']
# #         #                             )
                                    
# #         #                         elif v['function'] == 'geometry_consistency_loss':
# #         #                             val, _ = geometry_consistency_loss(p_points, g_points)
# #         #                             # 这里原代码似乎没把 val 加到 loss_dict? 如果不需要反向传播可忽略
                                    
# #         #                         elif v['function'] == 'normal_loss':
# #         #                             loss_dict[k], misc_dict[k] = normal_loss(p_points, g_points)
                                    
# #         #                         elif v['function'] == 'edge_loss':
# #         #                             loss_dict[k], misc_dict[k] = edge_loss(p_points, g_points)
                                    
# #         #                         elif v['function'] == 'normal_map_loss':
# #         #                             loss_dict[k], misc_dict[k] = normal_map_loss(p_normal, g_normal)
                                    
# #         #                         elif v['function'] == 'mask_bce_loss':
# #         #                             # mask loss 需要 float 类型的输入
# #         #                             loss_dict[k], misc_dict[k] = mask_bce_loss(p_mask, g_mask_fin, g_mask_inf)
                                    
# #         #                         elif v['function'] == 'mask_l2_loss':
# #         #                             loss_dict[k], misc_dict[k] = mask_l2_loss(p_mask, g_mask_fin, g_mask_inf)
                                    
# #         #                         elif v['function'] == 'multi_scale_gradient_loss':
# #         #                             loss_dict[k] = multi_scale_gradient_loss(
# #         #                                 p_points.unsqueeze(0), g_points.unsqueeze(0), 
# #         #                                 params=v.get('params', {})
# #         #                             )
# #         #                             misc_dict[k] = loss_dict[k]
                                    
# #         #                         elif v['function'] == 'metric_scale_loss':
# #         #                             if is_metric[i] and p_metric_scale is not None:
# #         #                                 loss_dict[k], misc_dict[k] = metric_scale_loss(p_metric_scale, gt_metric_scale)
# #         #                         else:
# #         #                             raise ValueError(f'Undefined loss function: {v["function"]}')
                            
# #         #                     # 后续的加权求和逻辑保持不变...
# #         #                     weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
# #         #                     loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                            
# #         #                     loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
# #         #                     loss_list.append(loss_)
                            
# #         #                     if isinstance(loss_, torch.Tensor) and torch.isnan(loss_).item():
# #         #                         pbar.write(f'NaN loss in process {accelerator.process_index}')

# #         #                     safe_misc_dict = {}
# #         #                     misc_dict = flatten_nested_dict(misc_dict)
# #         #                     for k_misc, v_misc in misc_dict.items():
# #         #                         k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
# #         #                         if isinstance(v_misc, torch.Tensor):
# #         #                             safe_misc_dict[k_misc_str] = v_misc.detach().float().item() 
# #         #                         else:
# #         #                             safe_misc_dict[k_misc_str] = v_misc

# #         #                     safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
# #         #                     records.append({**safe_loss_dict, **safe_misc_dict})
# #         #             # with torch.autocast(device_type=device.type, enabled=False):
# #         #             # loss_list, weight_list = [], []
# #         #             # for i in range(current_batch_size):
# #         #             #     # ================= 🛡️ [Fix 3] 防御性 Key 检查 =================
# #         #             #     current_label = str(label_type[i])
# #         #             #     if current_label not in config['loss']:
# #         #             #         continue
# #         #             #     # ==========================================================

# #         #             #     gt_metric_scale = None
# #         #             #     loss_dict, weight_dict, misc_dict = {}, {}, {}

# #         #             #     for k, v in config['loss'][current_label].items():
# #         #             #         weight_dict[k] = v['weight']
# #         #             #         if v['function'] == 'affine_invariant_global_loss':
# #         #             #             loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(pred_points[i], gt_points[i], **v['params'])
# #         #             #         elif v['function'] == 'affine_invariant_local_loss':
# #         #             #             loss_dict[k], misc_dict[k] = affine_invariant_local_loss(
# #         #             #                     pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), 
# #         #             #                     gt_focal[i], gt_metric_scale, **v['params'])
# #         #             #         elif v['function'] == 'geometry_consistency_loss':
# #         #             #             val, _ = geometry_consistency_loss(pred_points[i], gt_points[i])
# #         #             #         elif v['function'] == 'normal_loss':
# #         #             #             loss_dict[k], misc_dict[k] = normal_loss(pred_points[i], gt_points[i])
# #         #             #         elif v['function'] == 'edge_loss':
# #         #             #             loss_dict[k], misc_dict[k] = edge_loss(pred_points[i], gt_points[i])
# #         #             #         elif v['function'] == 'normal_map_loss':
# #         #             #             loss_dict[k], misc_dict[k] = normal_map_loss(pred_normal[i], gt_normal[i])
# #         #             #         elif v['function'] == 'mask_bce_loss':
# #         #             #             loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
# #         #             #         elif v['function'] == 'mask_l2_loss':
# #         #             #             loss_dict[k], misc_dict[k] = mask_l2_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
# #         #             #         elif v['function'] == 'multi_scale_gradient_loss':
# #         #             #             loss_dict[k] = multi_scale_gradient_loss(pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), params=v.get('params', {}))
# #         #             #             misc_dict[k] = loss_dict[k]
# #         #             #         elif v['function'] == 'metric_scale_loss':
# #         #             #             if is_metric[i] and pred_metric_scale is not None:
# #         #             #                 loss_dict[k], misc_dict[k] = metric_scale_loss(pred_metric_scale[i], gt_metric_scale)
# #         #             #         else:
# #         #             #             raise ValueError(f'Undefined loss function: {v["function"]}')
                        
# #         #             #     weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
# #         #             #     loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                        
# #         #             #     loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
# #         #             #     loss_list.append(loss_)
                        
# #         #             #     if isinstance(loss_, torch.Tensor) and torch.isnan(loss_).item():
# #         #             #         pbar.write(f'NaN loss in process {accelerator.process_index}')

# #         #             #     safe_misc_dict = {}
# #         #             #     misc_dict = flatten_nested_dict(misc_dict)
# #         #             #     for k_misc, v_misc in misc_dict.items():
# #         #             #         k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
# #         #             #         if isinstance(v_misc, torch.Tensor):
# #         #             #             safe_misc_dict[k_misc_str] = v_misc.detach().item() 
# #         #             #         else:
# #         #             #             safe_misc_dict[k_misc_str] = v_misc

# #         #             #     safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
# #         #             #     records.append({**safe_loss_dict, **safe_misc_dict})


# #         #             # 🔥 [Fix 4: Ghost Loss] 处理全 Batch 无效的情况
# #         #             if len(loss_list) > 0:
# #         #                 loss = sum(loss_list) / len(loss_list)
# #         #             # else:
# #         #             #     if accelerator.is_main_process:
# #         #             #         pbar.write(f"⚠️ [Step {i_step}] Entire batch invalid! Using dummy loss.")
# #         #             #     # 构造关联梯度的 0 值，保持计算图连接
# #         #             #     loss = pred_points.sum() * 0.0 
# #         #             else:
# #         #                 if accelerator.is_main_process:
# #         #                     pbar.write(f"⚠️ [Step {i_step}] Entire batch invalid! Using dummy loss.")
                        
# #         #                 # 使用 mean() 并过滤 nan，或者干脆用一个全新的 dummy tensor
# #         #                 # 方法 A: 试图挽救梯度图（推荐）
# #         #                 # 找到 pred_points 里合法的数值求和，乘以 0
# #         #                 valid_mask = torch.isfinite(pred_points)
# #         #                 if valid_mask.any():
# #         #                     loss = pred_points[valid_mask].sum() * 0.0
# #         #                 else:
# #         #                     # 如果全是 NaN，创建一个带梯度的 0 标量
# #         #                     loss = torch.tensor(0.0, device=device, requires_grad=True)
                            
# #         #                     # 🚨 并在此时手动清空梯度，防止之前的 NaN 梯度累积
# #         #                     optimizer.zero_grad()
# #         #             if torch.isnan(loss) or torch.isinf(loss):
# #         #                 print(f"⚠️ [Step {i_step}] Loss is NaN/Inf ({loss.item()})! Skipping backward.")
# #         #                 optimizer.zero_grad()
# #         #                 continue
# #         #             accelerator.backward(loss)

# #         #             if accelerator.sync_gradients:
# #         #                 # 🔥 [Fix 5: NaN 熔断] 防止梯度爆炸
# #         #                 params_to_check = [p for p in model.parameters() if p.grad is not None]
# #         #                 if len(params_to_check) > 0:
# #         #                     grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
# #         #                     if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                
# #         #                         if accelerator.is_main_process:
# #         #                             pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping.")
# #         #                         optimizer.zero_grad()
# #         #                         continue
                        
# #         #                 optimizer.step()
# #         #                 optimizer.zero_grad()

# #         #         i_accumulate += 1

# #         #     lr_scheduler.step()

# #             # if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
# #             #     if accelerator.distributed_type == accelerate.DistributedType.FSDP:
# #             #         from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# #             #         with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
# #             #             ema_model.update_parameters(model)
# #             #     else:
# #             #         ema_model.update_parameters(model)

# #             if i_step == initial_step or i_step % log_every == 0:
# #                 records = [key_average(records)]
# #                 records = accelerator.gather_for_metrics(records, use_gather_object=True)
# #                 if accelerator.is_main_process:
# #                     records = [
# #                         {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in r.items()}
# #                         for r in records
# #                     ]
# #                     records = key_average(records)
# #                     if enable_mlflow:
# #                         try:
# #                             mlflow.log_metrics(records, step=i_step)
# #                         except Exception as e:
# #                             print(f'Error while logging metrics to mlflow: {e}')
# #                 records = []

# #             # Save checkponits ... (保持原样)
# #             if accelerator.is_main_process and (i_step % save_every == 0):
# #                 pbar.write(f'Save checkpoint: {i_step:08d}')
# #                 Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)
                
# #                 with io.BytesIO() as f:
# #                     torch.save({
# #                         'model_config': config['model'],
# #                         'model': accelerator.unwrap_model(model).state_dict(),
# #                     }, f)
# #                     checkpoint_bytes = f.getvalue()
# #                 save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), checkpoint_bytes)

# #                 with io.BytesIO() as f:
# #                     torch.save({
# #                         'model_config': config['model'],
# #                         'step': i_step,
# #                         'optimizer': optimizer.state_dict(),
# #                         'lr_scheduler': lr_scheduler.state_dict(),
# #                     }, f)
# #                     checkpoint_bytes = f.getvalue()
# #                 save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'), checkpoint_bytes)
                
# #                 if enable_ema:
# #                     with io.BytesIO() as f:
# #                         torch.save({
# #                             'model_config': config['model'],
# #                             'model': ema_model.module.state_dict(),
# #                         }, f)
# #                         checkpoint_bytes = f.getvalue()
# #                     save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'), checkpoint_bytes)

# #                 with io.BytesIO() as f:
# #                     torch.save({
# #                         'model_config': config['model'],
# #                         'step': i_step,
# #                     }, f)
# #                     checkpoint_bytes = f.getvalue()
# #                 save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes)
            
# #             # Visualize ... (保持原样)
# #             if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0):
# #                 unwrapped_model = accelerator.unwrap_model(model)
# #                 save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
# #                 save_dir.mkdir(parents=True, exist_ok=True)
# #                 with torch.inference_mode():
# #                     for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
# #                         image, gt_depth, gt_intrinsics = batch['image'], batch['depth'], batch['intrinsics']
# #                         image, gt_depth, gt_intrinsics = image.to(device), gt_depth.to(device), gt_intrinsics.to(device)
# #                         output = unwrapped_model.infer(image)
# #                         pred_points = output['points'].cpu().numpy() if 'points' in output else None
# #                         pred_depth = output['depth'].cpu().numpy() if 'depth' in output else None
# #                         pred_mask =  output['mask'].cpu().numpy() if 'mask' in output else None
# #                         pred_normal = output['normal'].cpu().numpy() if 'normal' in output else None
# #                         image = (image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
# #                         for i_instance in range(image.shape[0]):
# #                             idx = i_batch * batch_size_forward + i_instance
# #                             save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
# #                             cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image[i_instance], cv2.COLOR_RGB2BGR))
# #                             if pred_points is not None:
# #                                 cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(pred_points[i_instance], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
# #                             if pred_depth is not None:
# #                                 cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(pred_depth[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
# #                             if pred_normal is not None:
# #                                 cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal_vis.png')), cv2.cvtColor(colorize_normal(pred_normal[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
            
# #             pbar.set_postfix({'loss': loss.item()}, refresh=False)
# #             pbar.update(1)

# # if __name__ == '__main__':
# #     main()

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
    
#     # 自动适配混合精度
#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         mixed_precision=None,
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

#     # ================= 🛡️ FSDP 策略 =================
#     if accelerator.distributed_type == accelerate.DistributedType.FSDP:
#         try:
#             if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
#                 block_module = model.encoder.backbone.blocks[0]
#                 block_class = type(block_module)
#                 import functools
#                 from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
#                 auto_wrap_policy = functools.partial(
#                     transformer_auto_wrap_policy,
#                     transformer_layer_cls={block_class},
#                 )
#                 accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
#         except Exception as e:
#             pass
#     # =================================================

#     # ================= 🟢 冻结 ViT Backbone & 启用 BN 保护 🟢 =================
#     print("❄️ Configuring Freeze/Train modes...")
    
#     # 1. 冻结 Backbone 参数
#     frozen_count = 0
#     if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
#         for param in model.encoder.backbone.parameters():
#             param.requires_grad = False
#             frozen_count += 1
#         print(f"✅ ViT Backbone frozen successfully! ({frozen_count} tensors frozen)")
#         model.encoder.backbone.eval() 
#     else:
#         print("⚠️ Warning: Could not find 'model.encoder.backbone' to freeze.")

#     # 2. 开启训练模式 (为了 Neck 和 Head)
#     model.train()

#     # 3. 强制冻结所有 BN 层的统计量 (Running Mean/Var)
#     def freeze_bn_stats(m):
#         if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm, nn.SyncBatchNorm)):
#             m.eval() 

#     print("🛡️ Applying BN Freeze Protection to the WHOLE model...")
#     model.apply(freeze_bn_stats)
#     # =================================================================================

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
#                     checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
#                 if 'optimizer' not in checkpoint and (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
#                     checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
#                 if enable_ema and accelerator.is_main_process:
#                     if 'ema_model' not in checkpoint and (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
#                         checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
#             else:
#                 print(f'No latest checkpoint found. Start from scratch.')
#                 checkpoint = None
#         else:
#             i_step = int(checkpoint_path)
#             checkpoint = {'step': i_step}
#             if (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
#                 checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
#             if (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
#                 checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
#             if enable_ema and accelerator.is_main_process:
#                 if (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
#                     checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']

#     if checkpoint is None:
#         print('Initialize model weights')
#         with accelerator.local_main_process_first():
#             model.init_weights()
#         initial_step = 0
#     else:
#         model.load_state_dict(checkpoint['model'], strict=False)
#         initial_step = checkpoint.get('step', 0) + 1
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
    
#     # 🟢 再次确认 BN 状态
#     model.apply(freeze_bn_stats)
#     nan_log_file = Path(workspace).joinpath("nan_images_log.txt")

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

#         if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
#             # ... (可视化 GT 代码省略，保持不变) ...
#             pass

#         if seed is not None:
#             set_seed(seed + initial_step, device_specific=True)   
        
#         # Training loop
#         for i_step in range(initial_step, num_iterations):

#             i_accumulate = 0
            
#             while i_accumulate < gradient_accumulation_steps:
#                 try:
#                     batch = train_data_pipe.get()
#                 except Exception as e:
#                     print(f"⚠️ Dataloader Error: {e}")
#                     continue

#                 image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = \
#                     batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']
                
#                 image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = \
#                     image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                
#                 current_batch_size = image.shape[0]
                
#                 gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                
#                 # 🛡️ [防爆修改 1] 安全的 Focal 计算
#                 # 防止 fx/fy 为 0 导致除以 0 产生 Inf
#                 fx = gt_intrinsics[..., 0, 0].float()
#                 fy = gt_intrinsics[..., 1, 1].float()
#                 epsilon = 1e-6
#                 gt_focal = (1.0 / torch.sqrt(1.0 / (fx ** 2 + epsilon) + 1.0 / (fy ** 2 + epsilon))).float()

#                 with accelerator.accumulate(model):
#                     if i_step <= config.get('low_resolution_training_steps', 0):
#                         num_tokens = config['model']['num_tokens_range'][0]
#                     else:
#                         num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
#                     # 混合精度 Forward
#                     autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
#                     is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
#                     with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
#                         output = model(image, num_tokens=num_tokens)
                    
#                     pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])
                    
#                     # 🛡️ [防爆修改 2] 逐样本 Loss 清洗流程
#                     # 强制退出 FP16，使用 FP32 计算 Loss，避免溢出
#                     with torch.autocast(device_type=device.type, enabled=False):
#                         valid_loss_list = []  # 存活下来的 Loss
                        
#                         # 🔄 逐样本细粒度检查 (Per-sample Check)
#                         for i in range(current_batch_size):
#                             # 1. 基础数据准备
#                             current_label = str(label_type[i])
#                             if current_label not in config['loss']: continue

#                             # 转换为 FP32 防止溢出
#                             p_points = pred_points[i].float()
#                             p_normal = pred_normal[i].float() if pred_normal is not None else None
#                             p_mask = pred_mask[i].float() if pred_mask is not None else None
#                             p_metric_scale = pred_metric_scale[i].float() if pred_metric_scale is not None else None

#                             g_points = gt_points[i].float()
#                             g_normal = gt_normal[i].float()
#                             g_mask_fin = gt_mask_fin[i]
#                             g_mask_inf = gt_mask_inf[i]
#                             g_focal = gt_focal[i] # 已经是 float 了

#                             gt_metric_scale = None 
#                             loss_dict, weight_dict, misc_dict = {}, {}, {}
#                             is_sample_broken = False # 标记当前样本是否损坏
                            
#                             # 2. 计算各项 Loss
#                             for k, v in config['loss'][current_label].items():
#                                 weight_dict[k] = v['weight']
                                
#                                 term_loss = torch.tensor(0.0, device=device) # 临时变量

#                                 if v['function'] == 'affine_invariant_global_loss':
#                                     term_loss, misc_dict[k], gt_metric_scale = affine_invariant_global_loss(p_points, g_points, **v['params'])
#                                 elif v['function'] == 'affine_invariant_local_loss':
#                                     term_loss, misc_dict[k] = affine_invariant_local_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), g_focal, gt_metric_scale, **v['params'])
#                                 elif v['function'] == 'geometry_consistency_loss':
#                                     val, _ = geometry_consistency_loss(p_points, g_points)
#                                     term_loss = val # 假设这里需要反向传播
#                                 elif v['function'] == 'normal_loss':
#                                     term_loss, misc_dict[k] = normal_loss(p_points, g_points)
#                                 elif v['function'] == 'edge_loss':
#                                     term_loss, misc_dict[k] = edge_loss(p_points, g_points)
#                                 elif v['function'] == 'normal_map_loss':
#                                     term_loss, misc_dict[k] = normal_map_loss(p_normal, g_normal)
#                                 elif v['function'] == 'mask_bce_loss':
#                                     term_loss, misc_dict[k] = mask_bce_loss(p_mask, g_mask_fin, g_mask_inf)
#                                 elif v['function'] == 'mask_l2_loss':
#                                     term_loss, misc_dict[k] = mask_l2_loss(p_mask, g_mask_fin, g_mask_inf)
#                                 elif v['function'] == 'multi_scale_gradient_loss':
#                                     term_loss = multi_scale_gradient_loss(p_points.unsqueeze(0), g_points.unsqueeze(0), params=v.get('params', {}))
#                                     misc_dict[k] = term_loss
#                                 elif v['function'] == 'metric_scale_loss':
#                                     if is_metric[i] and p_metric_scale is not None:
#                                         term_loss, misc_dict[k] = metric_scale_loss(p_metric_scale, gt_metric_scale)
                                
#                                 loss_dict[k] = term_loss
                                
#                                 # 🔥 立即检查单个 Loss 项是否 NaN/Inf
#                                 if torch.isnan(term_loss) or torch.isinf(term_loss):
#                                     is_sample_broken = True
#                                     # 不需要 break，继续算完其他项只是浪费一点点计算量，但逻辑更简单
                            
#                             # 如果单个 Loss 组件坏了，直接跳过当前样本
#                             if is_sample_broken:
#                                 bad_info = info[i]
#                                 if accelerator.is_main_process:
#                                     print(f"⚠️ [Step {i_step}] Skip Bad Image (NaN Component): {bad_info.get('dataset')}/{bad_info.get('filename')}")
#                                 continue 

#                             # 3. 聚合当前样本的总 Loss
#                             weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
#                             loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                            
#                             sample_total_loss = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
                            
#                             # 🛡️ [防爆修改 3] Loss 最终安检
#                             if torch.isnan(sample_total_loss) or torch.isinf(sample_total_loss):
#                                 bad_info = info[i]
#                                 if accelerator.is_main_process:
#                                     print(f"⚠️ [Step {i_step}] Skip Bad Image (Total NaN): {bad_info.get('dataset')}/{bad_info.get('filename')}")
#                                 continue
                            
#                             # 🛡️ [防爆修改 4] 数值截断 (Clamping)
#                             # 如果 Loss 巨大但不是 NaN (例如 10000.0)，截断到 100.0，防止梯度爆炸
#                             if sample_total_loss > 1000.0:
#                                 if accelerator.is_main_process:
#                                     print(f"⚠️ [Step {i_step}] Loss Spike ({sample_total_loss.item():.2f}) -> Clamped to 100.0")
#                                 sample_total_loss = torch.clamp(sample_total_loss, max=100.0)

#                             # ✅ 通过检查，加入有效列表
#                             valid_loss_list.append(sample_total_loss)
                            
#                             # 记录 Log
#                             safe_misc_dict = {}
#                             misc_dict = flatten_nested_dict(misc_dict)
#                             for k_misc, v_misc in misc_dict.items():
#                                 k_misc_str = '.'.join(k_misc) if isinstance(k_misc, tuple) else k_misc
#                                 if isinstance(v_misc, torch.Tensor):
#                                     safe_misc_dict[k_misc_str] = v_misc.detach().float().item() 
#                                 else:
#                                     safe_misc_dict[k_misc_str] = v_misc
#                             safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
#                             records.append({**safe_loss_dict, **safe_misc_dict})

#                         # 4. Batch 级聚合与 Backward
#                         if len(valid_loss_list) > 0:
#                             # 只对有效的样本求平均
#                             loss = sum(valid_loss_list) / len(valid_loss_list)
                            
#                             accelerator.backward(loss)
                            
#                             # 5. 梯度裁剪 (带 NaN 检查)
#                             if accelerator.sync_gradients:
#                                 params_to_check = [p for p in model.parameters() if p.grad is not None]
#                                 if len(params_to_check) > 0:
#                                     grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
#                                     if torch.isnan(grad_norm) or torch.isinf(grad_norm):
#                                         if accelerator.is_main_process:
#                                             pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping update.")
#                                         optimizer.zero_grad(set_to_none=True)
#                                     else:
#                                         optimizer.step()
#                                         optimizer.zero_grad(set_to_none=True)
#                                 else:
#                                     optimizer.step()
#                                     optimizer.zero_grad(set_to_none=True)
#                             else:
#                                 # 非 Sync 步 (Accumulation 阶段)
#                                 pass 
                            
#                         else:
#                             # 整个 Batch 全军覆没 (全是坏图)
#                             if accelerator.is_main_process:
#                                 pbar.write(f"⚠️ [Step {i_step}] All images in batch were invalid! Using dummy loss.")
                            
#                             # 构造 Dummy Loss 维持 DDP 及其同步
#                             # 必须让所有进程都执行 backward，否则 DDP 会 hang 住
#                             dummy_loss = pred_points.sum() * 0.0
#                             accelerator.backward(dummy_loss)
#                             optimizer.zero_grad(set_to_none=True)

#                 i_accumulate += 1

#             lr_scheduler.step()
            
#             # EMA Update
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

#             # Save checkponits
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
            
#             # Visualize
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
            
#             pbar.set_postfix({'loss': loss.item() if isinstance(loss, torch.Tensor) else loss}, refresh=False)
#             pbar.update(1)

# if __name__ == '__main__':
#     main()