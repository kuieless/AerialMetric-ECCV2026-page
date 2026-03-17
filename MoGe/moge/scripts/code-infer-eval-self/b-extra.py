import os
import shutil
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================

# 1. 定义任务列表
# 格式: {"name": "场景名", "input": "推理结果所在目录", "output": "提取后存放目录"}
# 根据你之前的 infer_batch.py 设置，路径如下：

TASKS = [
    {
        "name": "campus",
        "input": "/data1/szq/self1/campus",          # 推理输出的根目录
        "output": "/data1/szq/self1/extracted/campus" # 提取后的存放目录 (会自动创建)
    },
    {
        "name": "farm",
        "input": "/data1/szq/self1/farm",
        "output": "/data1/szq/self1/extracted/farm"
    },
    {
        "name": "gress",
        "input": "/data1/szq/self1/gress",
        "output": "/data1/szq/self1/extracted/gress"
    },
    {
        "name": "factory",
        "input": "/data1/szq/self1/factory",
        "output": "/data1/szq/self1/extracted/factory"
    },
]

# 2. 要提取的文件名
# MoGe 默认生成的是 'depth.npy'。如果你想提取点云，可以改成 'points.exr'
TARGET_FILENAME = "depth.npy" 

# 3. 线程数 (机械硬盘建议 4-8，固态硬盘建议 16-32)
NUM_THREADS = 16

# ===========================================

def copy_single_file(args):
    """ 单个文件复制任务 (用于多线程) """
    src_path, dst_path = args
    try:
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        return False

def process_scene(task):
    """ 处理单个场景 """
    scene_name = task["name"]
    input_dir = task["input"]
    output_dir = task["output"]
    
    print(f"\n📂 正在处理场景: {scene_name}")
    print(f"   源目录: {input_dir}")
    print(f"   目标目录: {output_dir}")

    if not os.path.exists(input_dir):
        print(f"   ❌ 错误: 源目录不存在，跳过。")
        return 0, 0

    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 2. 扫描所有待复制的文件
    # MoGe 输出结构通常是: input_dir / 图片名文件夹 / depth.npy
    copy_tasks = []
    
    # 获取 input_dir 下的所有子文件夹 (每个子文件夹代表一张图)
    try:
        subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    except Exception as e:
        print(f"   ❌ 读取目录失败: {e}")
        return 0, 0

    print(f"   🔍 扫描到 {len(subdirs)} 个图片文件夹，正在准备复制任务...")

    for subdir in subdirs:
        # 源文件: .../campus/DSC001/depth.npy
        src_file = os.path.join(input_dir, subdir, TARGET_FILENAME)
        
        # 目标文件: .../extracted/campus/DSC001.npy (重命名为图片名)
        # 注意：这里提取 subdir 的名字作为新的文件名
        dst_filename = f"{subdir}.npy" # 如果提取 exr，记得改成 .exr
        if TARGET_FILENAME.endswith('.exr'):
             dst_filename = f"{subdir}.exr"

        dst_file = os.path.join(output_dir, dst_filename)

        if os.path.exists(src_file):
            copy_tasks.append((src_file, dst_file))
    
    if not copy_tasks:
        print("   ⚠️ 未找到任何目标文件 (可能是文件名不匹配或推理未完成)")
        return 0, 0

    # 3. 多线程执行复制
    success_count = 0
    fail_count = 0
    
    # 使用线程池加速 IO 操作
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(tqdm(
            executor.map(copy_single_file, copy_tasks), 
            total=len(copy_tasks), 
            desc=f"   🚀 复制中 ({scene_name})",
            unit="file"
        ))
    
    success_count = sum(results)
    fail_count = len(results) - success_count

    print(f"   ✅ 完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count

def run_batch_extraction():
    print(f"{'='*60}")
    print(f"🛠️  批量提取脚本启动")
    print(f"🎯 目标文件: {TARGET_FILENAME}")
    print(f"🧵 线程数: {NUM_THREADS}")
    print(f"{'='*60}")

    total_success = 0
    total_fail = 0
    start_time = time.time()

    for task in TASKS:
        s, f = process_scene(task)
        total_success += s
        total_fail += f

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 全部处理完毕!")
    print(f"⏱  总耗时: {total_time:.2f} 秒")
    print(f"📊 总成功: {total_success}")
    print(f"📉 总失败: {total_fail}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_batch_extraction()