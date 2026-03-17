# CUDA_VISIBLE_DEVICES=5 python batch_infer.py
import os
import subprocess
import sys

# ================= 配置区域 =================

# 1. infer.py 脚本的绝对路径
SCRIPT_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/infer-syn.py"

# 2. 预训练模型的绝对路径
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"
MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/workspace/Infer6-large-max-throughput2-12.12-vitl-1-loss2.0-2/checkpoint/00004500.pt"


# 3. 新数据集的根目录 (搜索此目录下的文件夹)
DATASET_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data2-down"

# 4. 输出结果的根目录
# 结果将保存在这里，例如: .../Test-data-results/city-output_fov64
OUTPUT_ROOT = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data-results-4.5k"

# 5. 自动生成任务列表
TASKS = []

# 检查根目录是否存在
if os.path.exists(DATASET_ROOT):
    # 获取目录下所有文件/文件夹，并按名称排序
    # 例如: city-output_fov64, city-output_fov74
    all_scenes = sorted(os.listdir(DATASET_ROOT))
    
    print(f"📂 正在扫描目录: {DATASET_ROOT}")
    
    for scene_name in all_scenes:
        scene_path = os.path.join(DATASET_ROOT, scene_name)
        
        # 逻辑：
        # 1. 必须是文件夹
        # 2. 必须包含 'rgbs' 子文件夹
        if os.path.isdir(scene_path):
            input_rgbs_path = os.path.join(scene_path, "rgbs")
            
            if os.path.exists(input_rgbs_path) and os.path.isdir(input_rgbs_path):
                
                # 构建输出路径: OUTPUT_ROOT/场景名
                output_path = os.path.join(OUTPUT_ROOT, scene_name)
                
                # 创建任务字典
                task = {
                    "input": input_rgbs_path,   # 输入指向 rgbs 文件夹
                    "output": output_path,
                    "version": "v2",
                    "maps": True,   # 保存深度图等
                    "glb": False,
                    "fp16": True,   # 显存优化
                    # "resize": 1024 # 如果需要统一尺寸，取消注释
                }
                
                TASKS.append(task)
                print(f"➕ 已添加任务: {scene_name}")
                print(f"   └─ 输入: {input_rgbs_path}")
            else:
                # 只是调试信息，如果不需要看到跳过的文件夹，可以注释掉下面这行
                # print(f"⚪ 跳过: {scene_name} (未找到 rgbs 文件夹)")
                pass

else:
    print(f"❌ 错误: 数据集根目录不存在: {DATASET_ROOT}")

# ================= 执行逻辑 (保持不变) =================

def run_batch():
    # 设置环境变量，确保能处理 EXR
    env = os.environ.copy()
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    total = len(TASKS)
    
    if total == 0:
        print("\n⚠️ 没有发现符合条件的任务。")
        print(f"请检查路径: {DATASET_ROOT}")
        print("确保子文件夹中包含 'rgbs' 目录。")
        return

    print(f"\n🚀 开始批量推理，共 {total} 个任务...")
    print(f"📂 输出目录: {OUTPUT_ROOT}\n")

    for i, task in enumerate(TASKS):
        print(f"[任务 {i+1}/{total}] -----------------------------")
        print(f"场景: {os.path.basename(task['output'])}")
        
        # 自动创建输出父目录 (如果不存在)
        if not os.path.exists(task['output']):
            os.makedirs(task['output'], exist_ok=True)

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
            # 这里的 capture_output=False 让原本脚本的进度条能显示出来
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
        print(f"❌ 错误: 找不到推理脚本: {SCRIPT_PATH}")
    elif not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件: {MODEL_PATH}")
    else:
        run_batch()