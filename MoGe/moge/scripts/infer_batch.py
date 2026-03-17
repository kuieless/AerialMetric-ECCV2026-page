

#  CUDA_VISIBLE_DEVICES=5 python /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/infer_batch.py
import os
import subprocess
import sys

# ================= 配置区域 =================

# 1. infer.py 脚本的绝对路径 (保持不变)
SCRIPT_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/infer.py"

# 2. 预训练模型的绝对路径 (保持不变)
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/model.pt"
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitb-normal.pt"
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vits-normal.pt"
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"
MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/workspace/Infer6-large-max-throughput2-12.12-vitl-1-loss2.0-2/checkpoint/00004500.pt"
# 3. 新数据集的根目录 (请修改这里！！！)
# 假设你的 dj3-images, hsd1-images 等文件夹都在下面这个路径里
# DATASET_ROOT = "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all" 
DATASET_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/High-outputs-ODMExport_Summary/vnice" 


# # 4. 输出结果的根目录 (请修改这里！！！)
# OUTPUT_ROOT = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/4scene-infer-4k5"
# 4. 输出结果的根目录 (请修改这里！！！)
OUTPUT_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/High-outputs-ODMExport_Summary/vnice"

# 5. 自动生成任务列表
TASKS = []

# 检查根目录是否存在
if os.path.exists(DATASET_ROOT):
    # 获取目录下所有文件/文件夹，并按名称排序
    all_items = sorted(os.listdir(DATASET_ROOT))
    
    for item in all_items:
        full_path = os.path.join(DATASET_ROOT, item)
        
        # 逻辑：
        # 1. 必须是文件夹
        # 2. 文件夹名字必须以 "-images" 结尾
        # 这样会自动忽略 "-npy" 文件夹
        if os.path.isdir(full_path) and item.endswith("-images"):
            
            # 生成场景名称，去掉 "-images" 后缀
            # 例如: "dj3-images" -> "dj3"
            scene_name = item.replace("-images", "")
            
            # 构建输出路径，通常我们希望输出文件夹叫 "dj3" 而不是 "dj3-images"
            output_path = os.path.join(OUTPUT_ROOT, scene_name)
            
            # 创建任务字典
            task = {
                "input": full_path,
                "output": output_path,
                "version": "v2",
                "maps": True,   # 保存深度图等
                "glb": False,
                "fp16": True,  # 如果显存够，建议设为 True 加速
                # "resize": 1024 # 如果需要统一尺寸，可以取消注释
            }
            
            TASKS.append(task)
            print(f"➕ 已添加任务: {scene_name} (输入: {item})")

else:
    print(f"❌ 错误: 数据集根目录不存在: {DATASET_ROOT}")

# ================= 执行逻辑 (保持不变) =================

def run_batch():
    # 设置环境变量，确保能处理 EXR
    env = os.environ.copy()
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    total = len(TASKS)
    
    if total == 0:
        print("⚠️ 没有发现符合条件的任务。请检查 DATASET_ROOT 路径以及文件夹是否以 '-images' 结尾。")
        return

    print(f"\n🚀 开始批量推理，共 {total} 个任务...")

    for i, task in enumerate(TASKS):
        print(f"\n[任务 {i+1}/{total}] -----------------------------")
        print(f"场景: {os.path.basename(task['output'])}")
        print(f"输入: {task['input']}")
        print(f"输出: {task['output']}")

        # 构建命令
        cmd = [
            "python", SCRIPT_PATH,
            "--input", task["input"],
            "--output", task["output"],
            "--pretrained", MODEL_PATH,
            "--version", task.get("version", "v2")
        ]

        # 处理布尔开关 (Flag) 参数
        if task.get("maps"): cmd.append("--maps")
        if task.get("glb"): cmd.append("--glb")
        if task.get("ply"): cmd.append("--ply")
        if task.get("fp16"): cmd.append("--fp16")
        if task.get("show"): cmd.append("--show")

        # 处理其他数值参数
        if "fov_x" in task:
            cmd.extend(["--fov_x", str(task["fov_x"])])
        if "resize" in task:
            cmd.extend(["--resize", str(task["resize"])])
        if "resolution_level" in task:
            cmd.extend(["--resolution_level", str(task["resolution_level"])])

        # 执行命令
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ 任务 {i+1} 完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 任务 {i+1} 失败，错误代码: {e.returncode}")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")

    print(f"\n🎉 所有任务处理完毕。")

if __name__ == "__main__":
    # 简单的路径检查
    if not os.path.exists(SCRIPT_PATH):
        print(f"错误: 找不到推理脚本: {SCRIPT_PATH}")
    elif not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件: {MODEL_PATH}")
    else:
        run_batch()