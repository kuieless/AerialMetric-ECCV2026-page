import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import *
import importlib
import importlib.util

import click

@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations.')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.option('--ratio', type=float, default=1.0, help='Sampling ratio (0.0-1.0). Default 1.0 (100%).')
# 🔥 新增 Batch Size 配置
@click.option('--batch_size', type=int, default=4, help='Batch size for faster evaluation.')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, config_path: str, oracle_mode: bool, output_path: Union[str, Path], dump_pred: bool, dump_gt: bool, ratio: float, batch_size: int):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    import utils3d

    from moge.test.baseline import MGEBaselineInterface
    from moge.test.dataloader import EvalDataLoaderPipeline
    from moge.test.metrics import compute_metrics
    from moge.utils.geometry_torch import intrinsics_to_fov
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module
    
    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MGEBaselineInterface] = getattr(module, 'Baseline')
    baseline : MGEBaselineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    
    # Iterate over the dataset
    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        filenames, metrics_list = [], []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar
        ):  
            total_len = len(eval_data_pipe)
            target_limit = int(total_len * ratio)
            
            batch_samples = [] # 🔥 批处理缓冲区
            
            for i in range(total_len):
                if i >= target_limit:
                    break
                    
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                batch_samples.append(sample)
                
                # 当累积达到 batch_size 或触碰结尾时，统一推理
                if len(batch_samples) == batch_size or i == min(total_len, target_limit) - 1:
                    images = [s['image'] for s in batch_samples]
                    gt_intrinsics = [s['intrinsics'] for s in batch_samples] if oracle_mode else None

                    # Inference
                    torch.cuda.synchronize()
                    with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                        if oracle_mode:
                            # 如果 Baseline 支持批量接口就调用
                            if hasattr(baseline, 'infer_batch_for_evaluation'):
                                preds = baseline.infer_batch_for_evaluation(images, gt_intrinsics)
                            else:
                                preds = [baseline.infer_for_evaluation(img, intr) for img, intr in zip(images, gt_intrinsics)]
                        else:
                            if hasattr(baseline, 'infer_batch_for_evaluation'):
                                preds = baseline.infer_batch_for_evaluation(images)
                            else:
                                preds = [baseline.infer_for_evaluation(img) for img in images]
                        torch.cuda.synchronize()

                    time_per_sample = timer.time / len(batch_samples)

                    # Compute metrics and Dump
                    for pred, samp in zip(preds, batch_samples):
                        metrics, misc = compute_metrics(pred, samp, vis=dump_pred or dump_gt)
                        metrics['inference_time'] = time_per_sample
                        metrics_list.append(metrics)

                        dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', samp['filename'].replace('.zip', ''))
                        if dump_pred:
                            dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                            image_cpu = samp['image']
                            cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image_cpu.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                            with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                                json.dump(metrics, f, indent=4)

                            if 'pred_points' in misc:
                                points = misc['pred_points'].cpu().numpy()
                                cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                            
                            if 'pred_depth' in misc:
                                depth = misc['pred_depth'].cpu().numpy()
                                if 'mask' in pred:
                                    mask = pred['mask'].cpu().numpy()
                                    depth = np.where(mask, depth, np.inf)
                                cv2.imwrite(str(dump_path / 'pred' / 'depth.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))

                            if 'mask' in pred:
                                mask = pred['mask'].cpu().numpy()
                                cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (mask * 255).astype(np.uint8))

                            if 'normal' in pred:
                                normal = pred['normal'].cpu().numpy()
                                cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                            if 'intrinsics' in pred:
                                intrinsics = pred['intrinsics']
                                fov_x, fov_y = intrinsics_to_fov(intrinsics)
                                with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                                    json.dump({
                                        'fov_x': np.rad2deg(fov_x.item()),
                                        'fov_y': np.rad2deg(fov_y.item()),
                                        'intrinsics': intrinsics.cpu().numpy().tolist(),
                                    }, f)
                        
                        if dump_gt:
                            dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                            image_cpu = samp['image']
                            cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image_cpu.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                            if 'points' in samp:
                                points = samp['points']
                                cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                            if 'depth' in samp:
                                depth = samp['depth']
                                mask = samp['depth_mask']
                                cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                            if 'normal' in samp:
                                normal = samp['normal']
                                cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                            if 'depth_mask' in samp:
                                mask = samp['depth_mask']
                                cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))

                            if 'intrinsics' in samp:
                                intrinsics = samp['intrinsics']
                                fov_x, fov_y = intrinsics_to_fov(intrinsics)
                                with open(dump_path / 'gt' / 'info.json', 'w') as f:
                                    json.dump({
                                        'fov_x': np.rad2deg(fov_x.item()),
                                        'fov_y': np.rad2deg(fov_y.item()),
                                        'intrinsics': intrinsics.cpu().numpy().tolist(),
                                    }, f)

                    # Update progress bar
                    pbar.update(len(batch_samples))
                    
                    # Save intermediate results frequently to avoid losing data
                    if i % min(100, batch_size * 5) < batch_size or i == min(total_len, target_limit) - 1:
                        Path(output_path).write_text(
                            json.dumps({
                                **all_metrics, 
                                benchmark_name: key_average(metrics_list)
                            }, indent=4)
                        )
                    
                    # 清空批处理缓冲区
                    batch_samples = []

            all_metrics[benchmark_name] = key_average(metrics_list)

    # Save final results
    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))


if __name__ == '__main__':
    main()
