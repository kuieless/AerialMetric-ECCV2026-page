import os
import glob
import argparse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def resize_image(args):
    """
    单个图片缩放函数
    """
    file_path, output_dir, target_width, align_to = args
    
    try:
        filename = os.path.basename(file_path)
        save_path = os.path.join(output_dir, filename)

        with Image.open(file_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            w, h = img.size

            # 1. 计算缩放比例
            scale = target_width / float(w)
            
            # 2. 计算初步的目标高度
            raw_target_h = h * scale
            
            # 3. 强制对齐逻辑 (Round to nearest multiple of align_to)
            # 例如: align_to = 14
            # 新的宽度 = target_width (如果用户输入的target_width本身不是14倍数，这里最好也修整一下)
            new_w = int(round(target_width / align_to) * align_to)
            new_h = int(round(raw_target_h / align_to) * align_to)

            # 防止尺寸归零
            new_w = max(align_to, new_w)
            new_h = max(align_to, new_h)

            # 4. 执行高质量缩放 (LANCZOS 是下采样最好的选择)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 保存
            img_resized.save(save_path, quality=95)
            
        return None

    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="批量高质量下采样图片，并强制对齐倍数")
    parser.add_argument('--input', type=str, required=True, help='输入路径')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    parser.add_argument('--width', type=int, default=640, help='目标宽度 (默认: 640)')
    parser.add_argument('--align', type=int, default=14, help='强制对齐倍数 (默认: 14)')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='CPU核心数')
    
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 支持常见格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.input, ext)))

    total_files = len(image_files)
    print(f"Found {total_files} images.")
    print(f"Target Width: ~{args.width} | Align to multiple of: {args.align}")
    print(f"Resampling Method: LANCZOS (Best for downsampling)")

    tasks = [(f, args.output, args.width, args.align) for f in image_files]

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(resize_image, tasks), total=total_files, unit="img"))
        for res in results:
            if res: errors.append(res)

    print(f"Done. Processed: {total_files}. Errors: {len(errors)}")
    if errors:
        print(errors[:5])

if __name__ == '__main__':
    main()