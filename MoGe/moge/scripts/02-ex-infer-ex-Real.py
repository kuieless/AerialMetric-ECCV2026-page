


import os
import shutil
import time
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 输入根目录 (你的推理结果输出目录)
BASE_INPUT_DIR = "/home/data1/szq/Megadepth/becnmarkdata2/Val-Results-moge2-1000step-head"

# 2. 输出根目录 (提取出来的 npy 存放地)
BASE_OUTPUT_DIR = "/home/data1/szq/Megadepth/becnmarkdata2/Val-Results-moge2-1000step-head-out"

# 3. 需要处理的场景列表
# FOLDERS = [
#      "city_ouputfov74","city_ouputfov84","city_ouputfov64", "factory-output_fov74", "park-output_fov64", "park-output_fov84", 
#     "school-output_fov74", "vnicefinal-output_fov64", "vnicefinal-output_fov84", 
#     "factory-output_fov64", "factory-output_fov84", "park-output_fov74", 
#     "school-output_fov64", "school-output_fov84", "vnicefinal-output_fov74"

# ]
FOLDERS = [
    "ainterval5_AMtown01_cropped_downsampled",
    "interval5_AMvalley03_cropped_downsampled",
    "L1",
    "park14",
    "ainterval5_AMtown02_cropped_downsampled",
    "interval5_HKairport01_cropped_downsampled",
    "lewis-output",
    "park5",
    "ainterval5_HKisland02_cropped_downsampled",
    "interval5_HKairport02_cropped_downsampled",
    "lfls",
    "park8",
    "Artsci",
    "interval5_HKairport03_cropped_downsampled",
    "lfls2",
    "park9",
    "aukerman-output",
    "interval5_HKairport_GNSS01_cropped_downsampled",
    "longhua",
    "polytech",
    "BC1",
    "interval5_HKairport_GNSS02_cropped_downsampled",
    "ODM1-output",
    "R-PHD-output",
    "BC2",
    "interval5_HKairport_GNSS03_cropped_downsampled",
    "ODM2-output",
    "sceneca-output",
    "bellus-output",
    "interval5_HKairport_GNSS_Evening_cropped_downsampled",
    "ODM32-output",
    "SMBU",
    "caliterra-output",
    "interval5_HKisland01_cropped_downsampled",
    "ODM34-output",
    "S-output",
    "D-output",
    "interval5_HKisland03_cropped_downsampled",
    "ODM3-output",
    "sziit",
    "hav",
    "interval5_HKisland_GNSS01_cropped_downsampled",
    "ODM6-output",
    "sztu",
    "interval5_AMtown03_cropped_downsampled",
    "interval5_HKisland_GNSS02_cropped_downsampled",
    "park0",
    "upper",
    "interval5_AMvalley01_cropped_downsampled",
    "interval5_HKisland_GNSS03_cropped_downsampled",
    "park10",
    "yingrenshi",
    "interval5_AMvalley02_cropped_downsampled",
    "interval5_HKisland_GNSS_Evening_cropped_downsampled",
    "park13",
 "polytech",
  "longhua",
]

# 4. 自动生成任务列表
BATCH_TASKS = [
    {
        "input_dir": os.path.join(BASE_INPUT_DIR, f),
        "output_dir": os.path.join(BASE_OUTPUT_DIR, f),
        "target_filename": "depth.npy"  # 你的 infer.py 生成的文件名
    }
    for f in FOLDERS
]

# ===========================================

def copy_npy_worker(input_dir, output_dir, target_filename="depth.npy", task_id=1):
    """
    核心搬运函数: 直接复制文件，不进行 numpy 读写，速度极快。
    """
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 2. 获取子目录 (每个图片是一个子目录)
    try:
        subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    except FileNotFoundError:
        print(f"  [Task {task_id}] ❌ 错误: 输入目录不存在 -> {input_dir}")
        return 0, 0

    success_count = 0
    fail_count = 0

    # 3. 遍历处理
    pbar = tqdm(subdirs, desc=f"任务 {task_id}: {os.path.basename(input_dir)}")
    
    for subdir_name in pbar:
        # 源文件路径: .../scene/image_name/depth.npy
        src_path = os.path.join(input_dir, subdir_name, target_filename)
        
        # 目标文件路径: .../output/scene/image_name.npy (扁平化)
        dst_path = os.path.join(output_dir, f"{subdir_name}.npy")
        
        # 检查源文件是否存在
        if not os.path.exists(src_path):
            # 备选方案：如果找不到 depth.npy，尝试找目录下唯一的 .npy 文件
            try:
                all_files = os.listdir(os.path.join(input_dir, subdir_name))
                npy_files = [f for f in all_files if f.endswith('.npy') and 'depth' in f]
                if npy_files:
                    src_path = os.path.join(input_dir, subdir_name, npy_files[0])
                else:
                    fail_count += 1
                    continue
            except:
                fail_count += 1
                continue

        try:
            # 🔥 核心：直接文件复制 (比 np.load 快得多)
            shutil.copy2(src_path, dst_path)
            success_count += 1
        except Exception as e:
            # print(f"复制失败: {src_path} -> {e}")
            fail_count += 1
            
        pbar.set_postfix({"成功": success_count, "失败": fail_count})

    return success_count, fail_count

def run_batch_processing():
    total_tasks = len(BATCH_TASKS)
    print(f"\n{'='*60}")
    print(f"🚀 开始批量提取 NPY: 共 {total_tasks} 个场景")
    print(f"{'='*60}\n")

    global_success = 0
    global_fail = 0
    start_time = time.time()

    for i, task in enumerate(BATCH_TASKS):
        idx = i + 1
        in_dir = task.get("input_dir")
        out_dir = task.get("output_dir")
        fname = task.get("target_filename", "depth.npy")

        print(f"▶ 正在执行 ({idx}/{total_tasks})")
        print(f"  源: {in_dir}")
        print(f"  至: {out_dir}")

        s_count, f_count = copy_npy_worker(in_dir, out_dir, fname, task_id=idx)
        
        global_success += s_count
        global_fail += f_count
        print(f"  ✔ 完成. 成功: {s_count}, 失败: {f_count}\n")

    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"🎉 全部完成!")
    print(f"⏱ 总耗时: {total_time:.2f} 秒")
    print(f"📊 总计提取: {global_success}")
    print(f"📉 总计丢失: {global_fail}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_batch_processing()