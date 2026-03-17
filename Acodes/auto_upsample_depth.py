import os
import glob
import argparse
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

def read_depth(path):
    """
    读取深度图，支持图像格式和 .npy 格式
    """
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.npz'):
        data = np.load(path)
        return data['arr_0'] # 假设只有一个数组
    else:
        # 读取图像格式 (png, tif)，使用 -1 (IMREAD_UNCHANGED) 保留 16-bit 深度信息
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def save_depth(path, data):
    """
    保存深度图，保持与输入一致的格式
    """
    if path.endswith('.npy'):
        np.save(path, data)
    else:
        # 如果是浮点数且要保存为图像，OpenCV可能会截断，这里假设用户知道自己在做什么
        # 通常建议科研数据存 .npy，可视化存 .png
        cv2.imwrite(path, data)

def upsample_task(args):
    pred_path, gt_dir, output_dir, method_name = args
    
    filename = os.path.basename(pred_path)
    name_stem = Path(pred_path).stem  # 去除后缀的文件名，如 "frame_01"
    
    try:
        # 1. 在 GT 目录中寻找对应的参考文件 (名字相同，后缀可能不同)
        # 例如 pred 是 frame_01.npy, gt 可能是 frame_01.png
        gt_candidates = glob.glob(os.path.join(gt_dir, f"{name_stem}.*"))
        
        if not gt_candidates:
            return f"Skipped: No matching GT found for {filename}"
        
        # 取第一个找到的作为参考 (通常只有一个)
        gt_ref_path = gt_candidates[0]
        
        # 2. 读取 GT 尺寸
        # 我们不需要加载整个 GT 数据（为了快），只需要读 Header，但 OpenCV 读 Header 也很快
        if gt_ref_path.endswith('.npy'):
            gt_shape = np.load(gt_ref_path, mmap_mode='r').shape
            target_h, target_w = gt_shape[:2]
        else:
            img = cv2.imread(gt_ref_path, cv2.IMREAD_UNCHANGED)
            if img is None: return f"Error reading GT: {gt_ref_path}"
            target_h, target_w = img.shape[:2]

        # 3. 读取预测深度图
        pred_depth = read_depth(pred_path)
        if pred_depth is None: return f"Error reading Pred: {pred_path}"

        # 4. 执行上采样
        # 选择插值方法
        interp_method = cv2.INTER_CUBIC if method_name == 'bicubic' else cv2.INTER_LINEAR
        
        # cv2.resize 接收 (Width, Height)
        # 注意：如果 pred_depth 是 (H, W)，resize 会自动处理
        resized_depth = cv2.resize(pred_depth, (target_w, target_h), interpolation=interp_method)

        # 5. 保存
        # 输出文件名保持与输入一致
        save_path = os.path.join(output_dir, filename)
        save_depth(save_path, resized_depth)

        return None

    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="根据GT尺寸批量上采样深度预测图")
    parser.add_argument('--pred', type=str, required=True, help='预测结果文件夹 (Low Res)')
    parser.add_argument('--gt', type=str, required=True, help='真值/原图文件夹 (High Res Reference)')
    parser.add_argument('--output', type=str, required=True, help='输出文件夹')
    parser.add_argument('--method', type=str, default='bicubic', choices=['bicubic', 'bilinear'], help='插值方法')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='CPU核心数')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 搜索预测文件 (支持 .npy 和常见图片格式)
    extensions = ['*.npy', '*.png', '*.tif', '*.tiff', '*.jpg']
    pred_files = []
    for ext in extensions:
        pred_files.extend(glob.glob(os.path.join(args.pred, ext)))

    total_files = len(pred_files)
    print(f"Found {total_files} prediction files.")
    print(f"Matching sizes from GT dir: {args.gt}")
    print(f"Interpolation: {args.method.upper()}")

    # 准备任务
    tasks = [(f, args.gt, args.output, args.method) for f in pred_files]

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(upsample_task, tasks), total=total_files, unit="img"))
        for res in results:
            if res: errors.append(res)

    print(f"Done. Errors/Skips: {len(errors)}")
    if errors:
        print("First 5 errors:")
        for e in errors[:5]: print(e)

if __name__ == '__main__':
    main()