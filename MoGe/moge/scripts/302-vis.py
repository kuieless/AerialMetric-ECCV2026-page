

# import numpy as np
# import cv2
# import argparse
# from pathlib import Path
# import sys
# import matplotlib
# import matplotlib.pyplot as plt
# from typing import Union, Optional

# # 设置 Matplotlib 后端，避免在无 GUI 环境下报错
# matplotlib.use('Agg')

# def apply_cmap_matplotlib(
#     data_norm: np.ndarray, 
#     cmap_name: str = 'Spectral', 
#     valid_mask: Optional[np.ndarray] = None
# ) -> np.ndarray:
#     """
#     使用 Matplotlib 的 colormap 将归一化数据 (0~1) 转换为 BGR 图像。
#     """
#     try:
#         cmap = matplotlib.colormaps[cmap_name]
#     except KeyError:
#         cmap = plt.get_cmap(cmap_name)

#     # 1. 应用 colormap (输出为 RGBA 0.0-1.0)
#     colored_rgba = cmap(data_norm)
    
#     # 2. 转为 0-255 并取 RGB
#     colored_rgb = (colored_rgba[..., :3] * 255).astype(np.uint8)
    
#     # 3. 处理无效区域 (黑色)
#     if valid_mask is not None:
#         colored_rgb[~valid_mask] = [0, 0, 0]
    
#     # 4. RGB 转 BGR (OpenCV 格式)
#     colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    
#     return colored_bgr

# def create_depth_colorbar(
#     height: int, 
#     width: int, 
#     min_val: float, 
#     max_val: float, 
#     cmap_name: str = 'Spectral'
# ) -> np.ndarray:
#     """
#     创建一个颜色标尺：
#     - 上方：大数值 (Max)
#     - 下方：小数值 (Min)
#     """
#     # 1. 创建渐变条 (从上到下)
#     # 图像坐标系：行0是顶部，行H是底部。
#     # 我们希望顶部显示大值(对应归一化1.0)，底部显示小值(对应归一化0.0)。
#     # 所以 linspace 从 1.0 降到 0.0
#     gradient = np.linspace(1, 0, height).reshape(height, 1)
    
#     # 扩展宽度
#     gradient = np.tile(gradient, (1, width))
    
#     # 应用颜色映射
#     color_bar_img = apply_cmap_matplotlib(gradient, cmap_name=cmap_name)

#     # 2. 创建文本画布
#     text_canvas = np.full((height, 140, 3), 255, dtype=np.uint8) # 背景白色
    
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_color = (0, 0, 0)
    
#     # 3. 绘制刻度
#     num_labels = 7
#     for i in range(num_labels):
#         p = i / (num_labels - 1)  # p从 0.0 (顶部) 到 1.0 (底部)
        
#         # 计算 Y 坐标
#         y = int(p * (height - 20)) + 10
        
#         # 计算对应的值
#         # 顶部(p=0) 是 max_val
#         # 底部(p=1) 是 min_val
#         val = max_val * (1 - p) + min_val * p
        
#         # 绘制文本
#         label_text = f"{val:.1f}m"
#         cv2.putText(text_canvas, label_text, (10, y + 5), font, 0.6, font_color, 2, cv2.LINE_AA)
        
#         # 绘制连接线
#         cv2.line(color_bar_img, (width-15, y), (width, y), (255, 255, 255), 2)
            
#     return np.hstack([color_bar_img, text_canvas])

# def process_depth_file(
#     input_file_path: Path, 
#     output_file_path: Path, 
#     min_perc: float, 
#     max_perc: float, 
#     max_dist: Union[float, None],
#     cmap_name: str = 'Spectral'
# ):
#     try:
#         # --- 1. 加载数据 ---
#         print(f"📄 正在加载: {input_file_path}")
#         depth_map = np.load(input_file_path)
#         depth_map = np.squeeze(depth_map)
        
#         if depth_map.ndim != 2:
#             print(f"❌ 错误: 数据形状不正确 {depth_map.shape}")
#             return False

#         # --- 2. 数据清洗与统计 ---
#         valid_mask = depth_map > 0
        
#         if max_dist is not None:
#             print(f"ℹ️ 应用最大距离阈值: {max_dist}m")
#             valid_mask = valid_mask & (depth_map < max_dist)
            
#         valid_depths = depth_map[valid_mask]

#         if valid_depths.size == 0:
#             print(f"⚠️ 警告: 无有效深度值。")
#             min_depth, max_depth = 0.0, 1.0
#         else:
#             min_depth = np.percentile(valid_depths, min_perc)
#             max_depth = np.percentile(valid_depths, max_perc)
#             if min_depth >= max_depth:
#                 max_depth = min_depth + 1.0

#         print(f"📊 深度范围 ({min_perc}%-{max_perc}%): {min_depth:.2f}m - {max_depth:.2f}m")

#         # --- 3. 归一化 ---
#         depth_norm = depth_map.copy()
        
#         # 截断
#         depth_norm = np.clip(depth_norm, min_depth, max_depth)
        
#         # 线性归一化到 0.0 - 1.0 (0=min, 1=max)
#         depth_norm = (depth_norm - min_depth) / (max_depth - min_depth)
        
#         # --- 4. 颜色映射 ---
#         colorized_depth = apply_cmap_matplotlib(
#             depth_norm, 
#             cmap_name=cmap_name, 
#             valid_mask=valid_mask
#         )

#         # --- 5. 添加 Colorbar ---
#         h, w, _ = colorized_depth.shape
#         bar_w = max(int(w * 0.1), 80)
        
#         # 这里传入同样的 min/max，函数内部会处理成“小值在下”
#         color_bar = create_depth_colorbar(h, bar_w, min_depth, max_depth, cmap_name)
        
#         combined_image = np.hstack([colorized_depth, color_bar])

#         # --- 6. 保存 ---
#         output_file_path.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(output_file_path), combined_image)
#         print(f"✅ 保存至: {output_file_path}")
#         return True

#     except Exception as e:
#         print(f"❌ 异常: {input_file_path.name} -> {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def main(args):
#     input_path = Path(args.input_path)
#     output_arg = Path(args.output_path) if args.output_path else None
#     min_perc, max_perc = args.percentiles
#     max_dist = args.max_dist
#     cmap_name = args.cmap

#     if min_perc >= max_perc:
#         print(f"❌ 错误: 百分位数设置无效 {min_perc} >= {max_perc}")
#         sys.exit(1)

#     if not input_path.exists():
#         print(f"❌ 错误: 路径不存在 -> {input_path}")
#         sys.exit(1)

#     # --- 逻辑分流: 单文件 vs 目录 ---
#     if input_path.is_file():
#         if input_path.suffix != '.npy':
#             print("❌ 仅支持 .npy 文件")
#             sys.exit(1)
            
#         final_output_file = output_arg if (output_arg and output_arg.suffix) else \
#                             (output_arg / f"{input_path.stem}_vis.png" if output_arg else \
#                              input_path.parent / f"{input_path.stem}_vis.png")
                             
#         process_depth_file(input_path, final_output_file, min_perc, max_perc, max_dist, cmap_name)

#     elif input_path.is_dir():
#         output_dir = output_arg or input_path.parent / f"{input_path.name}_visualizations"
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         npy_files = sorted(list(input_path.glob('*.npy')))
#         print(f"🔍 发现 {len(npy_files)} 个文件，输出目录: {output_dir}")
        
#         success = 0
#         for i, f in enumerate(npy_files):
#             print(f"\n[{i+1}/{len(npy_files)}]")
#             dst = output_dir / f"{f.stem}_vis.png"
#             if process_depth_file(f, dst, min_perc, max_perc, max_dist, cmap_name):
#                 success += 1
#         print(f"\n完成: {success}/{len(npy_files)}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="NPY深度图可视化工具 (数值小在下)")
#     parser.add_argument("input_path", type=str, help="输入文件或目录")
#     parser.add_argument("-o", "--output_path", type=str, help="输出路径")
#     parser.add_argument("--percentiles", nargs=2, type=float, default=[2.0, 98.0], help="归一化百分比范围")
#     parser.add_argument("--max_dist", type=float, default=None, help="最大距离截断")
#     parser.add_argument("--cmap", type=str, default="Spectral", help="Matplotlib colormap 名称")
    
#     args = parser.parse_args()
#     main(args)

# #     '''
# # # 示例1：处理单个文件（默认设置 2%-98%）
# # python 302-visdepth-single.py /path/to/my/000010_cleaned.npy -o ./my_single_result.png

# # # 示例2：处理整个目录（默认设置 2%-98%）
# # python 302-visdepth-single.py /path/to/my_depths -o ./all_my_visualizations

# # # 示例3：[新功能] 处理单个文件，设置最大距离为 1000m
# # python /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/302-visdepth-single.py /home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize10400-4-hav/hav/depth_npy/hav --max_dist 1000

# # # 示例4：[新功能] 处理整个目录，使用 5% 到 95% 的百分位
# # python 302-visdepth-single.py /path/to/my_depths --percentiles 5.0 95.0

# # # 示例5：[新功能] 处理整个目录，同时设置百分位和最大距离
# '''
# python /home/szq/moge2/MoGe/moge/scripts/302-vis.py /data1/szq/data/Val/S-output/depth \
#     --percentiles 0 100 \
#     --max_dist 3000.0

#      '''

import numpy as np
import cv2
import argparse
from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict

# 设置 Matplotlib 后端，避免在无 GUI 环境下报错
matplotlib.use('Agg')

def apply_cmap_matplotlib(
    data_norm: np.ndarray, 
    cmap_name: str = 'Spectral', 
    valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """使用 Matplotlib 的 colormap 将归一化数据 (0~1) 转换为 BGR 图像。"""
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except KeyError:
        cmap = plt.get_cmap(cmap_name)

    colored_rgba = cmap(data_norm)
    colored_rgb = (colored_rgba[..., :3] * 255).astype(np.uint8)
    
    if valid_mask is not None:
        colored_rgb[~valid_mask] = [0, 0, 0]
    
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    return colored_bgr

def create_depth_colorbar(
    height: int, 
    width: int, 
    min_val: float, 
    max_val: float, 
    cmap_name: str = 'Spectral'
) -> np.ndarray:
    """创建颜色标尺"""
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.tile(gradient, (1, width))
    color_bar_img = apply_cmap_matplotlib(gradient, cmap_name=cmap_name)

    text_canvas = np.full((height, 140, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)
    
    num_labels = 7
    for i in range(num_labels):
        p = i / (num_labels - 1)
        y = int(p * (height - 20)) + 10
        val = max_val * (1 - p) + min_val * p
        label_text = f"{val:.1f}m"
        cv2.putText(text_canvas, label_text, (10, y + 5), font, 0.6, font_color, 2, cv2.LINE_AA)
        cv2.line(color_bar_img, (width-15, y), (width, y), (255, 255, 255), 2)
            
    return np.hstack([color_bar_img, text_canvas])

def process_depth_file(
    input_file_path: Path, 
    output_file_path: Path, 
    min_perc: float, 
    max_perc: float, 
    max_dist: Union[float, None],
    cmap_name: str = 'Spectral',
    silent: bool = False # 新增：控制是否打印详细日志
):
    try:
        # --- 1. 加载数据 ---
        # 如果不是 silent 模式才打印详细信息
        if not silent:
            print(f"  -> 处理: {input_file_path.name}")
            
        depth_map = np.load(input_file_path)
        depth_map = np.squeeze(depth_map)
        
        if depth_map.ndim != 2:
            print(f"❌ 错误: {input_file_path.name} 形状不正确 {depth_map.shape}")
            return False

        # --- 2. 数据清洗与统计 ---
        valid_mask = depth_map > 0
        if max_dist is not None:
            valid_mask = valid_mask & (depth_map < max_dist)
            
        valid_depths = depth_map[valid_mask]

        if valid_depths.size == 0:
            if not silent: print(f"⚠️ 警告: {input_file_path.name} 无有效深度值。")
            min_depth, max_depth = 0.0, 1.0
        else:
            min_depth = np.percentile(valid_depths, min_perc)
            max_depth = np.percentile(valid_depths, max_perc)
            if min_depth >= max_depth:
                max_depth = min_depth + 1.0

        # --- 3. 归一化 ---
        depth_norm = depth_map.copy()
        depth_norm = np.clip(depth_norm, min_depth, max_depth)
        depth_norm = (depth_norm - min_depth) / (max_depth - min_depth)
        
        # --- 4. 颜色映射 ---
        colorized_depth = apply_cmap_matplotlib(
            depth_norm, 
            cmap_name=cmap_name, 
            valid_mask=valid_mask
        )

        # --- 5. 添加 Colorbar ---
        h, w, _ = colorized_depth.shape
        bar_w = max(int(w * 0.1), 80)
        color_bar = create_depth_colorbar(h, bar_w, min_depth, max_depth, cmap_name)
        combined_image = np.hstack([colorized_depth, color_bar])

        # --- 6. 保存 ---
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file_path), combined_image)
        return True

    except Exception as e:
        print(f"❌ 异常: {input_file_path.name} -> {e}")
        return False

def find_npy_groups(root_path: Path) -> Dict[Path, List[Path]]:
    """
    智能搜索：递归查找所有 .npy 文件，并按它们所在的父文件夹分组。
    返回: { 父文件夹路径: [npy文件列表], ... }
    """
    groups = {}
    # rglob('*') 递归查找所有文件
    # 然后过滤出 suffix 为 .npy 的
    all_npy = sorted(list(root_path.rglob('*.npy')))
    
    for p in all_npy:
        parent = p.parent
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(p)
    return groups

def main(args):
    input_path = Path(args.input_path)
    output_root_arg = Path(args.output_path) if args.output_path else None
    min_perc, max_perc = args.percentiles
    max_dist = args.max_dist
    cmap_name = args.cmap

    if min_perc >= max_perc:
        print(f"❌ 错误: 百分位数设置无效 {min_perc} >= {max_perc}")
        sys.exit(1)

    if not input_path.exists():
        print(f"❌ 错误: 路径不存在 -> {input_path}")
        sys.exit(1)

    # --- 逻辑分流 ---
    
    # CASE 1: 单个文件
    if input_path.is_file():
        if input_path.suffix != '.npy':
            print("❌ 仅支持 .npy 文件")
            sys.exit(1)
        
        # 确定输出路径
        if output_root_arg:
            if output_root_arg.suffix: # 如果用户指定了文件名 (比如 -o result.png)
                final_output = output_root_arg
            else: # 如果用户指定了目录 (比如 -o ./results)
                final_output = output_root_arg / f"{input_path.stem}_vis.png"
        else:
            final_output = input_path.parent / f"{input_path.stem}_vis.png"
            
        process_depth_file(input_path, final_output, min_perc, max_perc, max_dist, cmap_name)

    # CASE 2: 目录 (智能批量处理)
    elif input_path.is_dir():
        print(f"🔍 正在递归搜索 '{input_path}' 下的所有 npy 文件夹...")
        
        # 1. 查找并分组
        npy_groups = find_npy_groups(input_path)
        
        if not npy_groups:
            print("⚠️ 未找到任何 .npy 文件。")
            sys.exit(0)
            
        total_folders = len(npy_groups)
        total_files = sum(len(v) for v in npy_groups.values())
        print(f"✅ 找到 {total_folders} 个包含npy的文件夹，共 {total_files} 个文件。\n")
        
        folder_idx = 0
        for folder_path, files in npy_groups.items():
            folder_idx += 1
            print(f"📂 [{folder_idx}/{total_folders}] 处理文件夹: {folder_path}")
            
            # 2. 确定该组文件的输出目录
            if output_root_arg:
                # 策略A: 镜像结构
                # 计算当前文件夹相对于输入根目录的相对路径
                # 例如: input=/data, folder=/data/scene1/depth -> relative=scene1/depth
                try:
                    rel_path = folder_path.relative_to(input_path)
                except ValueError:
                    # 如果路径解析出问题，直接用文件夹名
                    rel_path = folder_path.name
                
                target_dir = output_root_arg / rel_path
            else:
                # 策略B: 就地新建 _vis 目录
                # 例如: /data/scene1/depth -> /data/scene1/depth_vis
                target_dir = folder_path.parent / f"{folder_path.name}_vis"

            print(f"   💾 输出至: {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)

            # 3. 批量处理该文件夹下的文件
            success_count = 0
            for f in files:
                dst = target_dir / f"{f.stem}_vis.png"
                # 传递 silent=True 以减少刷屏，只在出错时报错
                if process_depth_file(f, dst, min_perc, max_perc, max_dist, cmap_name, silent=True):
                    success_count += 1
            
            print(f"   ✅ 完成: {success_count}/{len(files)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NPY深度图批量可视化工具 (支持递归搜索)")
    parser.add_argument("input_path", type=str, help="输入根目录或文件")
    parser.add_argument("-o", "--output_path", type=str, help="输出根目录 (如果不填，将在源文件夹旁生成 _vis 文件夹)")
    parser.add_argument("--percentiles", nargs=2, type=float, default=[2.0, 98.0], help="归一化百分比范围")
    parser.add_argument("--max_dist", type=float, default=None, help="最大距离截断")
    parser.add_argument("--cmap", type=str, default="Spectral", help="Matplotlib colormap 名称")
    
    args = parser.parse_args()
    main(args)


'''

python /home/szq/moge2/MoGe/moge/scripts/302-vis.py /data1/szq/Val/Bench-ori/Cleaned_Dataset_Factory/depth/0deaef9d111478171325640573d7bbe243878cde.npy \
    --percentiles 0 100 \
    --max_dist 3000.0 \
    -o /data1/szq/data/Val_Vis_Results

'''