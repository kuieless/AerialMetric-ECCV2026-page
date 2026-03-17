# import os
# import glob
# from PIL import Image
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# import argparse

# def crop_image(args):
#     """
#     单个图片处理函数，用于多进程调用
#     """
#     file_path, output_dir, divisor = args
    
#     try:
#         filename = os.path.basename(file_path)
#         save_path = os.path.join(output_dir, filename)

#         with Image.open(file_path) as img:
#             # 转换图片格式，防止部分PNG/RGBA格式报错
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
            
#             w, h = img.size

#             # 核心逻辑：计算最接近且小于当前尺寸的 14 的倍数
#             # 也就是：原始尺寸 - (原始尺寸 % 14)
#             new_w = w - (w % divisor)
#             new_h = h - (h % divisor)

#             # 如果图片本身小于14像素，则跳过
#             if new_w == 0 or new_h == 0:
#                 return f"Skipped (too small): {filename}"

#             # 如果尺寸已经完美，直接保存（或者你可以选择copy文件）
#             if new_w == w and new_h == h:
#                 img.save(save_path, quality=95)
#                 return None

#             # 计算居中裁剪的坐标
#             # 左边界 = (原宽 - 新宽) / 2
#             left = (w - new_w) // 2
#             top = (h - new_h) // 2
#             right = left + new_w
#             bottom = top + new_h

#             # 执行裁剪
#             img_cropped = img.crop((left, top, right, bottom))
            
#             # 保存结果
#             img_cropped.save(save_path, quality=95)
            
#         return None

#     except Exception as e:
#         return f"Error processing {file_path}: {str(e)}"

# def main():
#     parser = argparse.ArgumentParser(description="批量自适应裁剪图片为指定倍数大小")
#     parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
#     parser.add_argument('--output', type=str, required=True, help='输出图片文件夹路径')
#     parser.add_argument('--divisor', type=int, default=14, help='裁剪倍数 (默认: 14)')
#     parser.add_argument('--workers', type=int, default=os.cpu_count(), help='使用的CPU核心数 (默认: 全部核心)')
    
#     args = parser.parse_args()

#     # 1. 检查输出目录
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)
#         print(f"Created output directory: {args.output}")

#     # 2. 获取所有图片文件 (支持 jpg, png, jpeg, bmp)
#     extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
#     image_files = []
#     for ext in extensions:
#         # recursive=True 可以搜索子文件夹，这里为了简单默认只搜索当前层，如需递归请修改
#         image_files.extend(glob.glob(os.path.join(args.input, ext)))

#     total_files = len(image_files)
#     print(f"Found {total_files} images. Starting processing with {args.workers} workers...")
#     print(f"Target logic: Resize to nearest multiple of {args.divisor}")

#     # 3. 准备任务参数
#     # 将参数打包成 tuple 列表传递给多进程
#     tasks = [(f, args.output, args.divisor) for f in image_files]

#     # 4. 多进程处理
#     errors = []
    
#     with ProcessPoolExecutor(max_workers=args.workers) as executor:
#         # 使用 tqdm 显示进度条
#         results = list(tqdm(executor.map(crop_image, tasks), total=total_files, unit="img"))

#         # 收集错误信息
#         for res in results:
#             if res:
#                 errors.append(res)

#     # 5. 总结
#     print("\n" + "="*30)
#     print(f"Processing Complete!")
#     print(f"Total processed: {total_files}")
#     print(f"Errors/Skips: {len(errors)}")
    
#     if errors:
#         print("Log of issues:")
#         for err in errors[:10]: # 只打印前10个错误
#             print(err)
#         if len(errors) > 10:
#             print(f"... and {len(errors) - 10} more.")

# if __name__ == '__main__':
#     main()
import os
import glob
import numpy as np  # 需要 pip install numpy
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse

def crop_pair(args):
    """
    单个文件对处理函数：同时裁剪图片和对应的 NPY 文件
    """
    img_path, img_out_dir, npy_in_dir, npy_out_dir, divisor = args
    
    try:
        filename = os.path.basename(img_path)
        file_stem = os.path.splitext(filename)[0] # 获取文件名，不带后缀 (e.g., "0001")
        
        # -------------------------------------------------
        # 1. 处理图片，计算裁剪坐标
        # -------------------------------------------------
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            w, h = img.size

            # 计算目标尺寸 (向下取整到 divisor 的倍数)
            new_w = w - (w % divisor)
            new_h = h - (h % divisor)

            if new_w == 0 or new_h == 0:
                return f"Skipped (too small): {filename}"

            # 计算裁剪坐标 (PIL格式: left, top, right, bottom)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            right = left + new_w
            bottom = top + new_h

            # 保存图片
            img_save_path = os.path.join(img_out_dir, filename)
            
            # 如果尺寸不需要改变
            if new_w == w and new_h == h:
                img.save(img_save_path, quality=95)
                # 标记 npy 也无需裁剪，直接复制还是读取保存？为了统一逻辑，下面统一读取保存
                is_perfect_size = True
            else:
                img_cropped = img.crop((left, top, right, bottom))
                img_cropped.save(img_save_path, quality=95)
                is_perfect_size = False

        # -------------------------------------------------
        # 2. 处理对应的 NPY 文件 (如果提供了路径)
        # -------------------------------------------------
        if npy_in_dir and npy_out_dir:
            npy_filename = file_stem + ".npy"
            npy_path = os.path.join(npy_in_dir, npy_filename)
            npy_save_path = os.path.join(npy_out_dir, npy_filename)

            if os.path.exists(npy_path):
                # 读取 NPY
                data = np.load(npy_path)
                
                # 校验尺寸 (Numpy shape 通常是 (H, W) 或 (H, W, C))
                # 注意：img.size 是 (W, H)，data.shape 是 (H, W, ...)
                ny_h, ny_w = data.shape[:2]
                
                if ny_w != w or ny_h != h:
                    return f"Warning: Dimension mismatch for {filename} ({w}x{h}) vs {npy_filename} ({ny_w}x{ny_h})"

                if is_perfect_size:
                    # 尺寸完美，直接保存 (或者可以使用 shutil.copy 提高速度)
                    np.save(npy_save_path, data)
                else:
                    # 执行裁剪 (Numpy 切片格式: [top:bottom, left:right])
                    # 注意：如果数据是多通道 (H, W, C)，这种切片方式依然有效
                    data_cropped = data[top:bottom, left:right]
                    np.save(npy_save_path, data_cropped)
            else:
                return f"Warning: NPY file missing for {filename}"

        return None

    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="同步裁剪图片和NPY文件为指定倍数大小")
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出图片文件夹路径')
    
    # 新增 NPY 相关参数（可选）
    parser.add_argument('--input_npy', type=str, default=None, help='(可选) 输入NPY文件夹路径')
    parser.add_argument('--output_npy', type=str, default=None, help='(可选) 输出NPY文件夹路径')
    
    parser.add_argument('--divisor', type=int, default=14, help='裁剪倍数 (默认: 14)')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='使用的CPU核心数')
    
    args = parser.parse_args()

    # 1. 检查目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created image output: {args.output}")
        
    if args.input_npy and args.output_npy:
        if not os.path.exists(args.output_npy):
            os.makedirs(args.output_npy)
            print(f"Created npy output: {args.output_npy}")
    elif (args.input_npy and not args.output_npy) or (not args.input_npy and args.output_npy):
        print("Error: If processing NPY, both --input_npy and --output_npy must be provided.")
        return

    # 2. 获取所有图片文件
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.input, ext)))

    total_files = len(image_files)
    print(f"Found {total_files} images.")
    if args.input_npy:
        print("Mode: Sync Crop (Image + NPY)")
    else:
        print("Mode: Image Crop Only")

    # 3. 准备任务参数
    # 格式: (img_path, img_out, npy_in, npy_out, divisor)
    tasks = [(f, args.output, args.input_npy, args.output_npy, args.divisor) for f in image_files]

    # 4. 多进程处理
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(crop_pair, tasks), total=total_files, unit="pair"))

        for res in results:
            if res:
                errors.append(res)

    # 5. 总结
    print("\n" + "="*30)
    print(f"Processing Complete!")
    print(f"Total processed: {total_files}")
    print(f"Errors/Warnings: {len(errors)}")
    
    if errors:
        print("Log of issues:")
        for err in errors[:10]:
            print(err)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more.")

if __name__ == '__main__':
    main()

    '''
python auto_crop.py --input "./raw_images" --output "./processed_images"

    '''