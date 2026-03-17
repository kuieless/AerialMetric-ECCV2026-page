import argparse
import os
import shutil
import time
from tqdm import tqdm

def setup_args():
    parser = argparse.ArgumentParser(description="Step 2: Extract NPY (Benchmark Flat Mode)")
    parser.add_argument("--input_dir", type=str, required=True, help="Step 1 的输出目录 (例如 .../raw_preds)")
    parser.add_argument("--output_dir", type=str, required=True, help="扁平化存放目录 (例如 .../npy_flat)")
    return parser.parse_args()

def main():
    args = setup_args()
    
    BASE_INPUT_DIR = args.input_dir
    BASE_OUTPUT_DIR = args.output_dir

    print(f"🚀 [Step 2] 开始提取 NPY...")
    print(f"   源目录: {BASE_INPUT_DIR}")
    print(f"   目标目录: {BASE_OUTPUT_DIR}")

    # 1. 自动定位 mixed_val 文件夹
    # 兼容两种情况：用户传的是 raw_preds，或者直接传了 raw_preds/mixed_val
    mixed_val_path = os.path.join(BASE_INPUT_DIR, "mixed_val")
    
    if os.path.isdir(mixed_val_path):
        work_dir = mixed_val_path
        print(f"📂 发现混合数据集目录: {mixed_val_path}")
    elif os.path.isdir(os.path.join(BASE_INPUT_DIR, "DIODE")) or os.path.isdir(os.path.join(BASE_INPUT_DIR, "KITTI")): 
        # 防御性编程：如果 Step 0 没用 mixed_val 而是保留了原名，这里可能会挂，但在当前流程下应该只会走上面
        print("⚠️ 警告: 未找到 mixed_val，将尝试直接扫描输入目录作为扁平目录...")
        work_dir = BASE_INPUT_DIR
    else:
        # 最后的尝试：也许用户直接指向了 mixed_val
        work_dir = BASE_INPUT_DIR

    # 2. 扫描所有子文件夹 (每个子文件夹代表一张图)
    # 结构: work_dir / 图片名 / depth.npy
    try:
        subdirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    except FileNotFoundError:
        print(f"❌ 错误: 目录不存在 {work_dir}")
        exit(1)

    if not subdirs:
        print(f"❌ 错误: 在 {work_dir} 下未发现任何结果文件夹！")
        exit(1)

    print(f"📊 待处理文件数: {len(subdirs)}")
    
    # 3. 创建输出目录
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 4. 执行提取
    success = 0
    fail = 0
    
    pbar = tqdm(subdirs, desc="Extracting")
    for folder_name in pbar:
        # folder_name 就是 Renamed_Image 的文件名 (不含后缀)
        # 例如: DIODE_001
        
        src_path = os.path.join(work_dir, folder_name, "depth.npy")
        dst_path = os.path.join(BASE_OUTPUT_DIR, f"{folder_name}.npy")
        
        if os.path.exists(src_path):
            try:
                # 直接复制，去掉了中间的目录层级
                shutil.copy2(src_path, dst_path)
                success += 1
            except Exception as e:
                # print(f"Copy error: {e}")
                fail += 1
        else:
            # 可能是推理失败的样本
            fail += 1
            
        pbar.set_postfix({"OK": success, "Fail": fail})

    print("-" * 60)
    print(f"🎉 提取完成！")
    print(f"✅ 成功: {success}")
    print(f"❌ 缺失: {fail}")
    print(f"📂 结果已保存在: {BASE_OUTPUT_DIR}")
    print("-" * 60)

if __name__ == "__main__":
    main()