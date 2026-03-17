import os
import subprocess
import sys

# ================= 1. 基础环境配置 =================

# 你的推理脚本路径 (请确保这里指向的是你正在使用的脚本，infer.py 或 infer_fast.py)
SCRIPT_PATH = "/home/szq/moge2/MoGe/moge/scripts/infer-self.py"  
# SCRIPT_PATH = "/home/szq/moge2/MoGe/moge/scripts/infer_fast.py" # 如果用加速版，改这里

# 模型路径
# MODEL_PATH = "/home/szq/moge2/MoGe/vitl-normal.pt"

MODEL_PATH = "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-2/checkpoint/00016500_ema.pt"

# ================= 2. 采样配置 (新增功能) =================

# 采样比例 (0.1 代表 10%，即每 10 张图取 1 张)
# 设置为 1.0 代表处理所有图片
SAMPLING_RATIO = 0.1 

# 计算步长 (Stride)
# 例如: 10% -> 1/0.1 = 10，即 [::10]
STRIDE = int(1 / SAMPLING_RATIO) if SAMPLING_RATIO < 1.0 else 1

# ================= 3. 任务配置区域 =================

MANUAL_TASKS = [
    {
        "input":  "/home/szq/moge2/DJI-self2-final/Cleaned_Dataset_Campus/image",
        "output": "/data1/szq/self1/campus",
    },
    {
        "input":  "/home/szq/moge2/DJI-self2-final/Cleaned_Dataset_Farm/image",
        "output": "/data1/szq/self1/farm",
    },
    {
        "input":  "/home/szq/moge2/DJI-self2-final/Cleaned_Dataset_Gress/image",
        "output": "/data1/szq/self1/gress",
    },
    {
        "input":  "/home/szq/moge2/DJI-self2-final/Cleaned_Dataset_Factory/image",
        "output": "/data1/szq/self1/factory",
    },
]

# ================= 4. 通用参数配置 =================

COMMON_PARAMS = {
    "version": "v2",
    "maps": True,
    "glb": False,
    "ply": False,
    "fp16": True,
    "resize": 1024,   # 建议开启 resize 以加速
    # "batch_size": 2 # 如果是用 infer_fast.py，记得开启这个
}

# ================= 5. 执行逻辑 =================

def run_batch():
    env = os.environ.copy()
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ 错误: 找不到推理脚本: {SCRIPT_PATH}")
        return

    total = len(MANUAL_TASKS)
    print(f"\n🚀 开始批量推理 (采样率: {SAMPLING_RATIO*100}%, 步长: {STRIDE})")

    for i, task_config in enumerate(MANUAL_TASKS):
        input_dir = task_config["input"]
        output_dir = task_config["output"]

        if not os.path.exists(input_dir):
            print(f"⚠️ [跳过] 路径不存在: {input_dir}")
            continue

        print(f"\n[任务 {i+1}/{total}] -----------------------------")
        print(f"📂 输入: {input_dir}")
        print(f"💾 输出: {output_dir}")

        cmd = [
            "python", SCRIPT_PATH,
            "--input", input_dir,
            "--output", output_dir,
            "--pretrained", MODEL_PATH,
            "--version", task_config.get("version", COMMON_PARAMS["version"]),
            # ⬇️ 关键修改：传入 stride 参数
            "--stride", str(STRIDE) 
        ]

        # 合并参数
        if task_config.get("maps", COMMON_PARAMS["maps"]): cmd.append("--maps")
        if task_config.get("glb", COMMON_PARAMS["glb"]): cmd.append("--glb")
        if task_config.get("ply", COMMON_PARAMS["ply"]): cmd.append("--ply")
        if task_config.get("fp16", COMMON_PARAMS["fp16"]): cmd.append("--fp16")
        
        # 数值参数
        resize_val = task_config.get("resize", COMMON_PARAMS.get("resize"))
        if resize_val: cmd.extend(["--resize", str(resize_val)])
        
        batch_val = task_config.get("batch_size", COMMON_PARAMS.get("batch_size"))
        if batch_val: cmd.extend(["--batch_size", str(batch_val)])

        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ 任务 {i+1} 完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 任务 {i+1} 失败 (Code {e.returncode})")
            # 如果是因为脚本不支持 --stride 报错，提示用户
            if e.returncode != 0:
                print("💡 提示: 如果报错 'no such option: --stride'，请务必执行第二步修改 infer.py！")
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    run_batch()