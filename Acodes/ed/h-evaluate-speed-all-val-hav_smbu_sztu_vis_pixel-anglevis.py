import numpy as np
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import re # 引入正则表达式库
import cv2 # 引入 OpenCV 用于可视化

# =================================================================================
# 配置区域
# =================================================================================
USE_MEDIAN_SCALING = False
BATCH_SIZE = 32
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 400

OUTPUT_FILENAME = "hav-down-finetrain-5angle-01400.txt"
PITCH_ANGLES = [0, 15, 30, 45, 60]

# --- 可视化配置 ---
SAVE_VISUALIZATIONS = True # (开关) 是否保存可视化图像
OUTPUT_VIS_DIR = "/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GT11scene-normalize10400-666-101500-no300" # (目录) 可视化图像的输出根目录
VIS_COLORMAP_PRED = cv2.COLORMAP_JET # (深度图) Colormap
# -------------------------

EVALUATION_PATHS = [
        # {
        #     'name': '11scene',
        #     'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize10400-2-4-2/images_downsampled/depth_npy/images_downsampled',
        #     'gt_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-down/npy'
        # },

        #        {
        #     'name': '11scene',
        #     'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize10400-4-01500-3k/images_downsampled/depth_npy/images_downsampled',
        #     'gt_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-down/npy'
        # },


        {
            'name': '11scene',
            'pred_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-Moge-out',
            'gt_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-down/npy'
        },
                # {
        #     'name': '11scene',
        #     'pred_dir': '/home/data1/szq/Megadepth/metric3D/D3-angle/GT11scene-normalize10400-3/images_downsampled/depth_npy/images_downsampled',
        #     'gt_dir': '/home/data1/szq/Megadepth/metric3D/D1/Metric3D/GES-Test3-down/npy'
        # },
]

# =================================================================================
# 优化部分 1: DataLoader & 元数据提取 (与之前一致)
# =================================================================================

FILENAME_PATTERN = re.compile(r'(.+?)-(\d+)-(\d{6})$') 

class PairedNpyDataset(Dataset):
    def __init__(self, pred_dir, gt_dir, scene_base_name):
        all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        self.pred_files = []
        self.gt_files = []
        self.metadata = [] 
        self.scene_base_name = scene_base_name

        for pred_path in all_pred_files:
            basename = os.path.splitext(os.path.basename(pred_path))[0]
            gt_filename = f"{basename}.npy"
            gt_path = os.path.join(gt_dir, gt_filename)
            
            match = FILENAME_PATTERN.match(basename)
            if not match:
                continue

            scene_name_part = match.group(1) 
            angle = int(match.group(2))

            if os.path.exists(gt_path):
                self.pred_files.append(pred_path)
                self.gt_files.append(gt_path)
                self.metadata.append({
                    'full_scene': self.scene_base_name,
                    'sub_scene': scene_name_part,
                    'angle': angle
                })
        
        if len(self.pred_files) == 0:
            raise ValueError(f"在 '{pred_dir}' 和 '{gt_dir}' 之间未找到任何匹配的文件对。")
        
        print(f"       -> 找到 {len(self.pred_files)} 个有效的 PRED/GT 文件对。")

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, idx):
        pred_np = np.load(self.pred_files[idx]).astype(np.float32)
        gt_np = np.load(self.gt_files[idx]).astype(np.float32)

        if pred_np.ndim == 3:
            pred_np = np.squeeze(pred_np)
        if gt_np.ndim == 3:
            gt_np = np.squeeze(gt_np)
            
        pred_tensor = torch.from_numpy(pred_np)
        gt_tensor = torch.from_numpy(gt_np)
        
        meta = self.metadata[idx].copy() 
        meta['basename'] = os.path.splitext(os.path.basename(self.pred_files[idx]))[0]
        
        return pred_tensor, gt_tensor, meta

def collate_fn_with_meta(batch):
    pred_tensors = [item[0] for item in batch]
    gt_tensors = [item[1] for item in batch]
    metadata = [item[2] for item in batch]
    
    return pred_tensors, gt_tensors, metadata

# =================================================================================
# 优化部分 2: 核心代码向量化 (与之前一致)
# =================================================================================

def compute_errors_torch(gt, pred, mask):
    B = gt.shape[0]
    
    valid_pixel_count = mask.sum(dim=[1, 2]).to(torch.float32)
    gt_clamped = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_clamped = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    thresh = torch.maximum((gt_clamped / pred_clamped), (pred_clamped / gt_clamped))
    
    a1_count = ((thresh < 1.25) & mask).sum(dim=[1, 2]).to(torch.float32)
    a2_count = ((thresh < 1.25 ** 2) & mask).sum(dim=[1, 2]).to(torch.float32)
    a3_count = ((thresh < 1.25 ** 3) & mask).sum(dim=[1, 2]).to(torch.float32)

    rmse_map = (gt - pred) ** 2
    rmse_map[~mask] = 0
    rmse_sum_sq = rmse_map.sum(dim=[1, 2])

    rmse_log_map = (torch.log(gt_clamped) - torch.log(pred_clamped)) ** 2
    rmse_log_map[~mask] = 0
    rmse_log_sum_sq = rmse_log_map.sum(dim=[1, 2])

    abs_rel_map = torch.abs(gt - pred) / gt_clamped
    abs_rel_map[~mask] = 0
    abs_rel_sum = abs_rel_map.sum(dim=[1, 2])

    sq_rel_map = ((gt - pred) ** 2) / gt_clamped
    sq_rel_map[~mask] = 0
    sq_rel_sum = sq_rel_map.sum(dim=[1, 2])

    results = torch.stack([
        abs_rel_sum, sq_rel_sum, rmse_sum_sq, rmse_log_sum_sq, 
        a1_count, a2_count, a3_count, valid_pixel_count
    ], dim=1)
    
    return results

def compute_metrics_from_sums(errors_sum_array):
    total_sums = errors_sum_array
    total_valid_pixels = total_sums[7]
    
    if total_valid_pixels == 0:
        return np.zeros(7)
    
    final_metrics = np.zeros(7)
    
    final_metrics[0] = total_sums[0] / total_valid_pixels
    final_metrics[1] = total_sums[1] / total_valid_pixels
    final_metrics[2] = np.sqrt(total_sums[2] / total_valid_pixels)
    final_metrics[3] = np.sqrt(total_sums[3] / total_valid_pixels)
    final_metrics[4] = total_sums[4] / total_valid_pixels
    final_metrics[5] = total_sums[5] / total_valid_pixels
    final_metrics[6] = total_sums[6] / total_valid_pixels
    
    return final_metrics

# =================================================================================
# 优化部分 2.5: 可视化函数 (已升级)
# =================================================================================

# --- (A) 预测深度图 (热力图) & 标尺 ---

def create_depth_colorbar_img(img_height, min_val, max_val, colormap_id):
    """
    (新增) 创建一个垂直的深度标尺图像。
    """
    bar_width = 30
    text_width = 100
    total_width = bar_width + text_width
    
    # 1. 创建颜色条
    # (注意: linspace(255, 0) 使 "热" (255) 在顶部, "冷" (0) 在底部)
    gradient = np.linspace(255, 0, img_height, dtype=np.uint8).reshape(-1, 1)
    color_bar = cv2.applyColorMap(gradient, colormap_id)
    color_bar = cv2.resize(color_bar, (bar_width, img_height))

    # 2. 创建文本背景
    text_bg = np.full((img_height, text_width, 3), 255, dtype=np.uint8) # 白色背景
    
    # 3. 添加文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0) # 黑色
    thickness = 1
    
    # 顶部 (Max)
    max_text = f"{max_val:.1f} m"
    cv2.putText(text_bg, max_text, (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    # 底部 (Min)
    min_text = f"{min_val:.1f} m"
    cv2.putText(text_bg, min_text, (10, img_height - 15), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    # 中间值
    mid_val = (max_val + min_val) / 2
    mid_text = f"{mid_val:.1f} m"
    cv2.putText(text_bg, mid_text, (10, img_height // 2 + 7), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # 4. 合并
    colorbar_img = np.hstack((color_bar, text_bg))
    return colorbar_img

def save_depth_colormap(pred_batch, mask_batch, meta):
    """
    (可视化 1, 已升级) 将预测深度图保存为带标尺的热力图。
    """
    try:
        pred_np = pred_batch[0].cpu().numpy()
        mask_np = mask_batch[0].cpu().numpy()
        
        min_val, max_val = 0, 1
        if mask_np.sum() > 0:
            min_val = np.min(pred_np[mask_np])
            max_val = np.max(pred_np[mask_np])
            if max_val == min_val:
                max_val += 1e-6
        
        # 归一化。 cv2.normalize 将 min_val 映射到 0 (冷), max_val 映射到 255 (热)
        normalized_depth = cv2.normalize(pred_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_map = cv2.applyColorMap(normalized_depth, VIS_COLORMAP_PRED)
        color_map[~mask_np] = [0, 0, 0] # 掩码外为黑色
        
        # (新增) 创建并附加标尺
        img_height = color_map.shape[0]
        colorbar = create_depth_colorbar_img(img_height, min_val, max_val, VIS_COLORMAP_PRED)
        
        combined_image = np.hstack((color_map, colorbar))
        
        # 保存
        vis_sub_dir = os.path.join(OUTPUT_VIS_DIR, meta['full_scene'], f"angle_{meta['angle']}")
        os.makedirs(vis_sub_dir, exist_ok=True)
        basename = meta['basename']
        output_path = os.path.join(vis_sub_dir, f"{basename}_pred_vis.png")
        
        cv2.imwrite(output_path, combined_image)
        
    except Exception as e:
        print(f"\n[警告] 保存 {meta['basename']} 的 *预测深度图* 可视化时出错: {e}")


# --- (B) a1/a2/a3 误差图 & 嵌入式图例 ---

# (B, G, R) 颜色定义
VIS_COLORS_A1A2A3 = {
    "PASS_A1": torch.tensor([255, 0, 0], dtype=torch.uint8),   # 蓝色 (最好)
    "PASS_A2": torch.tensor([0, 255, 0], dtype=torch.uint8),   # 绿色
    "PASS_A3": torch.tensor([0, 255, 255], dtype=torch.uint8), # 黄色
    "FAIL_A3": torch.tensor([0, 0, 255], dtype=torch.uint8),     # 红色 (最差)
    "INVALID": torch.tensor([128, 128, 128], dtype=torch.uint8) # 灰色 (无效区域)
}
# 阈值
THRESH_A1 = 1.25
THRESH_A2 = 1.25 ** 2
THRESH_A3 = 1.25 ** 3

def create_a1_legend_img():
    """
    (新增) 创建 a1/a2/a3 图例图像 (Numpy array)。
    """
    width = 400 # 增加宽度以便调整大小
    height_per_entry = 30
    num_entries = 5
    
    colors_np = {
        f"Pass a1 (< {THRESH_A1:.2f})": (255, 0, 0),       # Blue
        f"Pass a2 (< {THRESH_A2:.2f})": (0, 255, 0),       # Green
        f"Pass a3 (< {THRESH_A3:.2f})": (0, 255, 255),     # Yellow
        f"Fail a3 (>= {THRESH_A3:.2f})": (0, 0, 255),      # Red
        "Invalid / No GT": (128, 128, 128) # Gray
    }
    
    image = np.full((height_per_entry * num_entries, width, 3), 255, dtype=np.uint8) # 白色背景
    
    y_offset = 0
    for text, color in colors_np.items():
        cv2.rectangle(image, (10, y_offset + 5), (40, y_offset + 25), color, -1)
        cv2.putText(image, text, (55, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += height_per_entry
        
    return image

def save_a1_a2_a3_map(gt_batch, pred_batch, mask_batch, meta, device, legend_img):
    """
    (可视化 2, 已升级) 在GPU上生成 a1/a2/a3 误差图并附加图例后保存。
    """
    try:
        # 1. 确保颜色在正确的设备上
        for key in VIS_COLORS_A1A2A3:
            VIS_COLORS_A1A2A3[key] = VIS_COLORS_A1A2A3[key].to(device)

        # 2. 计算 Thresh Map (B=1, H, W)
        gt_clamped = torch.clamp(gt_batch, min=MIN_EVAL_DEPTH)
        pred_clamped = torch.clamp(pred_batch, min=MIN_EVAL_DEPTH)
        thresh_map = torch.maximum((gt_clamped / pred_clamped), (pred_clamped / gt_clamped))
        
        thresh_map_flat = thresh_map[0]
        mask_valid_flat = mask_batch[0]
        
        # 3. 创建空白画布 (H, W, 3)
        map_h, map_w = gt_batch.shape[1:]
        vis_image_gpu = torch.zeros(map_h, map_w, 3, dtype=torch.uint8, device=device)

        # 4. 填充颜色 (从最差到最好)
        vis_image_gpu[~mask_valid_flat] = VIS_COLORS_A1A2A3["INVALID"]
        mask = (thresh_map_flat >= THRESH_A3) & mask_valid_flat
        vis_image_gpu[mask] = VIS_COLORS_A1A2A3["FAIL_A3"]
        mask = (thresh_map_flat < THRESH_A3) & (thresh_map_flat >= THRESH_A2) & mask_valid_flat
        vis_image_gpu[mask] = VIS_COLORS_A1A2A3["PASS_A3"]
        mask = (thresh_map_flat < THRESH_A2) & (thresh_map_flat >= THRESH_A1) & mask_valid_flat
        vis_image_gpu[mask] = VIS_COLORS_A1A2A3["PASS_A2"]
        mask = (thresh_map_flat < THRESH_A1) & mask_valid_flat
        vis_image_gpu[mask] = VIS_COLORS_A1A2A3["PASS_A1"]
        
        # 5. 转移回 CPU
        vis_image_numpy = vis_image_gpu.cpu().numpy()

        # 6. (新增) 调整图例大小并附加
        # 将图例的宽度调整为与主图像一致，保持图例的高度
        legend_resized = cv2.resize(legend_img, (map_w, legend_img.shape[0]), interpolation=cv2.INTER_AREA)
        combined_image = np.vstack((vis_image_numpy, legend_resized))

        # 7. 保存图像
        vis_sub_dir = os.path.join(OUTPUT_VIS_DIR, meta['full_scene'], f"angle_{meta['angle']}")
        os.makedirs(vis_sub_dir, exist_ok=True)
        basename = meta['basename']
        output_path = os.path.join(vis_sub_dir, f"{basename}_a1_map.png")
        
        cv2.imwrite(output_path, combined_image)

    except Exception as e:
        print(f"\n[警告] 保存 {meta['basename']} 的 *a1/a2/a3图* 可视化时出错: {e}")

# (移除 generate_legend 函数，它已被 create_a1_legend_img 替代)

# =================================================================================
# 优化部分 3: 主循环集成 (已修改)
# =================================================================================

def evaluate_single_scene(pred_dir, gt_dir, scene_base_name, num_workers, 
                          use_median_scaling=True, batch_size=16, 
                          min_depth=1e-3, max_depth=80, 
                          legend_img=None): # (新增) 接收图例
    device = torch.device("cuda")
    try:
        dataset = PairedNpyDataset(pred_dir, gt_dir, scene_base_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_with_meta)
    except ValueError as e:
        print(f"警告: 场景 {scene_base_name} 初始化失败 ({e})。跳过此场景。")
        return None, None

    all_errors_scene = []
    all_metadata_scene = []
    pbar_desc = f"评估中: {scene_base_name}"
    
    for pred_list, gt_list, metadata_list in tqdm(dataloader, desc=pbar_desc, unit="batch", leave=False):
        
        for pred_tensor, gt_tensor, meta in zip(pred_list, gt_list, metadata_list):
            
            # 1. 数据传输到 GPU
            pred_batch = pred_tensor.to(device).unsqueeze(0) # (1, H, W)
            gt_batch = gt_tensor.to(device).unsqueeze(0)     # (1, H, W)
            
            # 2. 在 GPU 上进行插值
            target_shape = gt_batch.shape[2:] 
            if pred_batch.shape[2:] != target_shape:
                pred_batch = pred_batch.unsqueeze(1) 
                pred_batch = F.interpolate(pred_batch, size=target_shape, mode='bilinear', align_corners=False)
                pred_batch = pred_batch.squeeze(1)
            
            # 3. 创建掩码
            mask_batch = (gt_batch > min_depth) & (gt_batch < max_depth)
            
            if use_median_scaling:
                # Median Scaling
                gt_nan = gt_batch.clone()
                pred_nan = pred_batch.clone()
                gt_nan[~mask_batch] = torch.nan
                pred_nan[~mask_batch] = torch.nan
                median_gt = torch.nanmedian(gt_nan, dim=[1, 2]).values 
                median_pred = torch.nanmedian(pred_nan, dim=[1, 2]).values 
                ratio = median_gt / median_pred
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
                pred_batch = pred_batch * ratio.view(1, 1, 1)

            # 4. 裁剪预测值
            pred_batch.clamp_(min=min_depth, max=max_depth)
            
            # 5. 调用误差函数 (返回 (1, 8))
            errors_single = compute_errors_torch(gt_batch, pred_batch, mask_batch)
            
            # 6. (已修改) 生成并保存 *两种* 可视化
            if SAVE_VISUALIZATIONS:
                # (可视化 1) 预测深度热力图 (带标尺)
                save_depth_colormap(pred_batch, mask_batch, meta)
                
                # (可视化 2) a1/a2/a3 误差图 (带图例)
                if legend_img is not None:
                    save_a1_a2_a3_map(gt_batch, pred_batch, mask_batch, meta, device, legend_img)


            # 7. 收集结果 (转移回CPU)
            all_errors_scene.append(errors_single.cpu().numpy()[0]) # [0] 提取 (8,) 数组
            all_metadata_scene.append(meta) # 收集元数据

    errors_combined = np.stack(all_errors_scene, axis=0) if all_errors_scene else None
    
    return errors_combined, all_metadata_scene

def format_results(mean_errors, name):
    header = (" {:<65} |" + " {:>8} |" * 7).format(name, "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    line = ("-" * 67 + "|") + ("-" * 10 + "|") * 7
    values = (" {:<65} |" + "&{: 8.3f} |" * 7).format(name, *mean_errors.tolist())
    return f"{header}\n{line}\n{values}\n"

# =================================================================================
# 优化部分 4: 主函数 (已修改)
# =================================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA GPU。程序终止。")
        exit()
        
    # (新增) 在主函数开始时，只创建一次图例图像
    A1_LEGEND_IMG = None
    if SAVE_VISUALIZATIONS:
        print("正在创建 a1/a2/a3 图例模板...")
        A1_LEGEND_IMG = create_a1_legend_img()
        os.makedirs(OUTPUT_VIS_DIR, exist_ok=True) # 确保根目录存在
        
    all_results_by_type = {} 
    all_errors_combined = [] 

    print(f"开始进行多场景精细评估... 模式: {'有中位数定标' if USE_MEDIAN_SCALING else '无定标 (绝对尺度)'}")
    print(f"可视化: {'已启用 (保存带图例/标尺的图像至 ' + OUTPUT_VIS_DIR + ')' if SAVE_VISUALIZATIONS else '已禁用'}")
    print("-" * 50)

    for path_info in tqdm(EVALUATION_PATHS, desc="总进度", unit="scene"):
        scene_base_name = path_info['name']
        pred_dir = path_info['pred_dir']
        gt_dir = path_info['gt_dir']

        print(f"\n正在处理场景: {scene_base_name}")
        print(f" -> PRED路径: {pred_dir}")
        print(f" -> GT路径:   {gt_dir}")
        
        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            print(f"警告: 找不到场景 '{scene_base_name}' 的路径，已跳过。")
            continue

        scene_errors, scene_metadata = evaluate_single_scene(
            pred_dir, gt_dir, 
            scene_base_name,
            num_workers=NUM_WORKERS,
            use_median_scaling=USE_MEDIAN_SCALING,
            batch_size=BATCH_SIZE, 
            min_depth=MIN_EVAL_DEPTH,
            max_depth=MAX_EVAL_DEPTH,
            legend_img=A1_LEGEND_IMG # (新增) 传入图例
        )
        
        if scene_errors is not None:
            all_errors_combined.append(scene_errors)
            for error_sum_arr, meta in zip(scene_errors, scene_metadata):
                key_sub_scene = (meta['full_scene'], meta['sub_scene'], 'ALL') 
                all_results_by_type.setdefault(key_sub_scene, []).append(error_sum_arr)
                key_full_scene = (meta['full_scene'], 'ALL', 'ALL') 
                all_results_by_type.setdefault(key_full_scene, []).append(error_sum_arr)
                key_angle_group = ('ALL', 'ALL', meta['angle'])
                all_results_by_type.setdefault(key_angle_group, []).append(error_sum_arr)
                key_detailed = (meta['full_scene'], meta['sub_scene'], meta['angle'])
                all_results_by_type.setdefault(key_detailed, []).append(error_sum_arr)

    # 报告生成逻辑
    report_lines = []
    report_lines.append("=" * 145)
    report_lines.append(f"深度评估报告 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"评估模式: {'有中位数定标 (相对几何精度)' if USE_MEDIAN_SCALING else '无定标 (绝对米制尺度)'}")
    report_lines.append(f"评估方法: 像素加权平均 (Pixel-Weighted Average)")
    report_lines.append(f"可视化: {'已启用 (保存带图例/标尺的图像至 ' + OUTPUT_VIS_DIR + ')' if SAVE_VISUALIZATIONS else '已禁用'}")
    report_lines.append("=" * 145)

    
    # === 1. 总览结果 ===
    if all_errors_combined:
        all_errors_combined_np = np.concatenate(all_errors_combined, axis=0)
        overall_sums = all_errors_combined_np.sum(axis=0)
        overall_mean_metrics = compute_metrics_from_sums(overall_sums)
        report_lines.append(format_results(overall_mean_metrics, "===> 总平均 (Overall Average)"))
        report_lines.append("=" * 145)

    # === 2. 按俯仰角分组报告 ===
    report_lines.append("\n" + "=" * 145)
    report_lines.append("### 按俯仰角 (Pitch Angle) 聚合的平均结果 ###")
    report_lines.append("=" * 145)
    for angle in sorted(PITCH_ANGLES):
        key = ('ALL', 'ALL', angle)
        if key in all_results_by_type:
            angle_sums = np.stack(all_results_by_type[key], axis=0).sum(axis=0)
            angle_mean_metrics = compute_metrics_from_sums(angle_sums)
            name = f"所有场景 (ALL SCENES) - 俯仰角 {angle}°"
            report_lines.append(format_results(angle_mean_metrics, name))
            
    # === 3. 按原始场景名报告 ===
    report_lines.append("\n" + "=" * 145)
    report_lines.append("### 按原始场景名 (EVAL_PATH.name) 聚合的报告 ###")
    report_lines.append("=" * 145)
    
    for path_info in EVALUATION_PATHS:
        scene_name = path_info['name']
        key_overall = (scene_name, 'ALL', 'ALL')
        
        if key_overall in all_results_by_type:
            scene_sums = np.stack(all_results_by_type[key_overall], axis=0).sum(axis=0)
            scene_mean_metrics = compute_metrics_from_sums(scene_sums)
            report_lines.append(format_results(scene_mean_metrics, f"场景总平均: {scene_name}"))
            
            report_lines.append("\n" + "-" * 145)
            report_lines.append(f"场景 {scene_name} 的子场景精细报告:")
            report_lines.append("-" * 145)
            
            scene_specific_keys = [k for k in all_results_by_type.keys() if k[0] == scene_name]

            def custom_sort_key(key):
                sub_scene = key[1]
                angle_val = key[2]
                if angle_val == 'ALL':
                    angle_order = -1
                else:
                    angle_order = angle_val 
                return (sub_scene, angle_order)

            detailed_keys = sorted(scene_specific_keys, key=custom_sort_key)
            
            for d_key in detailed_keys:
                if d_key[1] == 'ALL': 
                    continue
                if d_key[2] == 'ALL':
                    sub_scene_sums = np.stack(all_results_by_type[d_key], axis=0).sum(axis=0)
                    sub_scene_mean_metrics = compute_metrics_from_sums(sub_scene_sums)
                    name = f" ├ 子场景总平均: {d_key[1]}"
                    report_lines.append(format_results(sub_scene_mean_metrics, name))
                elif d_key[2] in PITCH_ANGLES:
                    angle_sums = np.stack(all_results_by_type[d_key], axis=0).sum(axis=0)
                    angle_mean_metrics = compute_metrics_from_sums(angle_sums)
                    name = f" └── {d_key[1]} - 俯仰角 {d_key[2]}°"
                    report_lines.append(format_results(angle_mean_metrics, name))
            
            report_lines.append("-" * 145)
            
    # --- 最终报告输出 ---
    final_report = "\n".join(report_lines)
    print("\n\n" + final_report)
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(final_report)
        
    print(f"\n报告已成功保存到: {OUTPUT_FILENAME}")
    if SAVE_VISUALIZATIONS:
        print(f"可视化图像 (带嵌入式图例/标尺) 已保存到: {OUTPUT_VIS_DIR}")
    print("评估完成!")