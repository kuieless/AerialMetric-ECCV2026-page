import os
import argparse
from pathlib import Path

# 定义要统计的图片扩展名
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
# 定义要统计的NPY扩展名
NPY_EXTENSIONS = ['.npy']

def get_file_count(folder_path, valid_extensions):
    """统计文件夹中符合扩展名的文件数量"""
    if not os.path.isdir(folder_path):
        return -1 # 标记为文件夹不存在
    
    count = 0
    try:
        # 使用 os.scandir 遍历，比 os.listdir 快
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():
                    # 检查扩展名 (忽略大小写)
                    if any(entry.name.lower().endswith(ext) for ext in valid_extensions):
                        count += 1
    except Exception as e:
        print(f"Error reading {folder_path}: {e}")
        return 0
    return count

def scan_scenes(root_dir):
    root = Path(root_dir)
    if not root.exists():
        print(f"错误: 路径 '{root_dir}' 不存在")
        return

    print(f"正在统计文件数量: {root} ...\n")

    # 1. 自动发现场景名
    # 逻辑：寻找所有以 "-images" 结尾的文件夹，提取前面的部分作为场景名
    all_subdirs = [d for d in os.listdir(root) if os.path.isdir(root / d)]
    scene_names = set()
    
    for d in all_subdirs:
        if d.endswith("-images"):
            scene_names.add(d[:-7]) # 去掉 "-images"
        elif d.endswith("-npy"):
            scene_names.add(d[:-4]) # 去掉 "-npy" (以防只有npy没有image的情况)

    if not scene_names:
        print("未在该目录下发现符合 *-images 或 *-npy 命名规范的文件夹。")
        return

    # 2. 准备表格数据
    table_data = []
    total_scenes = len(scene_names)
    mismatch_scenes = 0

    for scene in sorted(list(scene_names)):
        img_dir = root / f"{scene}-images"
        npy_dir = root / f"{scene}-npy"

        # 统计数量
        n_img = get_file_count(img_dir, IMG_EXTENSIONS)
        n_npy = get_file_count(npy_dir, NPY_EXTENSIONS)

        # 格式化输出字符
        str_img = str(n_img) if n_img != -1 else "缺失"
        str_npy = str(n_npy) if n_npy != -1 else "缺失"

        # 判断状态
        if n_img == -1 or n_npy == -1:
            status = "⚠️ 文件夹缺失"
            mismatch_scenes += 1
        elif n_img == n_npy:
            if n_img == 0:
                status = "⚠️ 空文件夹"
            else:
                status = "✅ 匹配"
        else:
            status = f"❌ 数量不一 (差 {abs(n_img - n_npy)})"
            mismatch_scenes += 1

        table_data.append([scene, str_img, str_npy, status])

    # 3. 打印表格
    # 定义列宽
    col_fmt = "{:<25} | {:<15} | {:<15} | {:<20}"
    print("=" * 85)
    print(col_fmt.format("场景名称 (Scene)", "图片数量", "NPY数量", "状态"))
    print("-" * 85)

    for row in table_data:
        print(col_fmt.format(*row))

    print("=" * 85)
    print(f"统计完成。总场景数: {total_scenes}")
    if mismatch_scenes == 0:
        print("🎉 完美！所有场景的文件数量都一一对应。")
    else:
        print(f"⚠️ 注意：有 {mismatch_scenes} 个场景存在问题（缺失或数量不符）。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计数据集图像和NPY文件数量")
    # 默认路径设为您提供的路径，也可以通过命令行参数修改
    parser.add_argument("path", nargs="?", default="/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all", 
                        help="要扫描的根目录路径")
    
    args = parser.parse_args()
    scan_scenes(args.path)