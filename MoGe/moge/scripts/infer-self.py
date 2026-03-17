import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings
import matplotlib
matplotlib.use('Agg') # 必须加这一行，防止在无显示器的服务器上报错
import matplotlib.pyplot as plt
import click
import numpy as np

def save_depth_with_colorbar(depth, save_path, title="Depth", unit="m"):
    """
    保存带标尺的深度热力图
    """
    # 1. 过滤无效值并自动计算对比度范围 (2% - 98%)
    valid_mask = (depth > 0) & np.isfinite(depth)
    if valid_mask.sum() == 0:
        return # 空数据不保存
    
    valid_values = depth[valid_mask]
    vmin = np.percentile(valid_values, 2)
    vmax = np.percentile(valid_values, 98)
    
    # 2. 绘图
    plt.figure(figsize=(10, 8))
    # 使用 'Spectral' 色带 (MoGe 风格)
    cmap = plt.get_cmap('Spectral')
    cmap.set_bad(color='black') # 无效区域显示黑色
    
    # 截断显示范围以增强对比度
    depth_clamped = np.clip(depth, vmin, vmax)
    depth_clamped[~valid_mask] = np.nan
    
    img_plot = plt.imshow(depth_clamped, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # 3. 添加标尺
    cbar = plt.colorbar(img_plot, fraction=0.046, pad=0.04)
    cbar.set_label(f'{title} ({unit})', rotation=270, labelpad=20)
    
    plt.title(f"{title}\nRange: [{vmin:.2f}, {vmax:.2f}] {unit}")
    plt.axis('off')
    plt.tight_layout()
    
    # 4. 保存并关闭
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

@click.command(help='Inference script')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
@click.option('--output', '-o', 'output_path', default='./output', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2']), default='v2', help='Model version. Defaults to "v2"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for much faster inference.')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--threshold', type=float, default=0.04, help='Threshold for removing edges. Defaults to 0.01. Smaller value removes more edges. "inf" means no thresholding.')
# --- [修改点 1] 新增 stride 参数 ---
@click.option('--stride', type=int, default=1, help='Sampling stride (e.g. 10 means process 1 out of every 10 images).') 
# --------------------------------
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')

def main(
    input_path: str,
    fov_x_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    threshold: float,
    stride: int, # --- [修改点 2] 函数入口增加 stride ---
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):  
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import click

    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
    import utils3d
    device = torch.device(device_name)

    # 1. 扫描文件
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    # --- [修改点 3] 应用 stride 采样逻辑 ---
    if stride > 1:
        print(f"Applying stride: {stride} (Sampling 1/{stride})")
        original_count = len(image_paths)
        # Python 切片操作，每隔 stride 取一张
        image_paths = image_paths[::stride] 
        print(f"Images reduced from {original_count} to {len(image_paths)}")
    # -----------------------------------

    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    # 2. 加载模型
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = "Ruicheng/moge-2-vitl-normal" if model_version == "v2" else "Ruicheng/moge-vitl"
    
    print(f"Loading model from: {pretrained_model_name_or_path}")
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    
    if use_fp16:
        model.half()
        print("⚡ FP16 mode enabled.")

    # 3. 循环推理
    # 使用 torch.inference_mode() 关闭梯度，比 no_grad() 更快更省显存
    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc='Inference'):
            if not image_path.exists(): continue
            
            # --- 读取与预处理 ---
            # 直接读取 BGR
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Error reading {image_path}")
                continue
                
            # 保持 BGR 给 OpenCV 保存用，转 RGB 给模型用
            height, width = image_bgr.shape[:2]
            
            # 只有当设定了 resize 且尺寸不同时才缩放
            if resize_to is not None:
                scale = resize_to / max(height, width)
                if scale < 1.0:
                    new_w, new_h = int(width * scale), int(height * scale)
                    # 确保是 14 的倍数 (DINOv2 patch)
                    new_w = (new_w // 14) * 14
                    new_h = (new_h // 14) * 14
                    image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 转 RGB 并归一化
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).to(device)
            # 增加 batch 维度: (C, H, W) -> (1, C, H, W) 
            # (infer 内部通常处理 3D tensor, 但加上维度更保险，或者保持原样看 infer 实现)
            # MoGe 的 model.infer 接受 (C, H, W)

            # --- 模型推理 ---
            output = model.infer(
                image_tensor, 
                fov_x=fov_x_, 
                resolution_level=resolution_level, 
                num_tokens=num_tokens, 
                use_fp16=use_fp16
            )
            
            # --- 提取数据 (只拿需要的) ---
            # 深度图 (metric depth)
            depth = output['depth'].cpu().numpy()
            points = output['points'].cpu().numpy()
            # 尺度因子 (Scale)
            metric_scale_val = 1.0
            if 'metric_scale' in output:
                metric_scale_val = output['metric_scale'].float().cpu().item()

            # --- 保存结果 ---
            save_dir = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
            save_dir.mkdir(exist_ok=True, parents=True)

            # 1. 保存原图 (JPG)
            cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr)
            
            # 2. 保存深度数据 (NPY) - 最快，无损
            np.save(str(save_dir / 'depth.npy'), depth)

            # 3. [ADDED] Save Points (EXR)
            # Note: OpenCV expects BGR, so we convert RGB points to BGR before saving
            cv2.imwrite(
                str(save_dir / 'points.exr'), 
                cv2.cvtColor(points, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
            ) # <--- ADDED: Save points.exr
            
            # 3. 保存 Scale 信息 (JSON)
            with open(save_dir / 'scale.json', 'w') as f:
                json.dump({
                    'metric_scale': metric_scale_val,
                    'formula': 'metric_depth = relative_depth * metric_scale'
                }, f, indent=4)

            # --- [新增] 保存 FoV 信息 ---
            if 'intrinsics' in output:
                intrinsics = output['intrinsics'].cpu().numpy()
                # 计算 FoV
                fov_x_rad, fov_y_rad = utils3d.np.intrinsics_to_fov(intrinsics)
                fov_x_deg = float(np.rad2deg(fov_x_rad))
                fov_y_deg = float(np.rad2deg(fov_y_rad))
                
                # 打印日志
                # tqdm.write(f"  [{image_path.stem}] 最佳 FOV: {fov_x_deg:.2f}°") 

                # 保存 JSON
                with open(save_dir / 'fov.json', 'w') as f:
                    json.dump({
                        'fov_x': round(fov_x_deg, 2),
                        'fov_y': round(fov_y_deg, 2),
                    }, f, indent=4)

            # 4. 保存可视化图 (Matplotlib) 
            # ⚠️ 这是整个流程中最慢的一步，如果以后还嫌慢，请注释掉下面这行
            # save_depth_with_colorbar(
            #     depth, 
            #     save_dir / 'depth_vis_scaled.png', 
            #     title="Metric Depth", 
            #     unit="m"
            # )

if __name__ == '__main__':
    main()