import os
import sys
import subprocess
from pathlib import Path

# ================= ⚙️ 配置区域 =================

# 1. 想要评估的权重列表
CHECKPOINTS = [
# "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00000000_ema.pt",
# "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00001200_ema.pt",
# "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00002400_ema.pt",
# "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00003600_ema.pt",
# "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00004800_ema.pt",
    "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00006000_ema.pt",
    # 添加更多...
]

# 2. 输出根目录 (结果会保存在这个目录下的子文件夹里)
OUTPUT_ROOT = "/data1/szq/Table-head-ground"

# 3. 指定 GPU (对应 CUDA_VISIBLE_DEVICES)
GPU_ID = "4"

# 4. 采样率 (你之前修改了脚本支持 ratio，这里可以设置)
#    1.0 = 跑全量, 0.2 = 跑 20%
SAMPLING_RATIO = 0.2 

# 5. 项目路径配置
PROJECT_ROOT = "/home/szq/moge2/MoGe"
SCRIPT_REL_PATH = "moge/scripts/eval_baseline.py"  # 脚本相对于根目录的位置
BASELINE_CODE = "baselines/moge.py"
CONFIG_PATH = "configs/eval/all_benchmarks.json"

# ================= 🚀 执行逻辑 =================

def run_batch_eval():
    # 确保输出根目录存在
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 设置环境变量 (指定 GPU 和 PYTHONPATH)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    total_tasks = len(CHECKPOINTS)
    print(f"🔥 准备开始批量评估，共 {total_tasks} 个模型...")
    print(f"📍 项目根目录: {PROJECT_ROOT}")
    print(f"💾 输出目录: {OUTPUT_ROOT}")
    print(f"⚙️  GPU: {GPU_ID} | Ratio: {SAMPLING_RATIO}")

    for i, ckpt_path in enumerate(CHECKPOINTS):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"❌ 找不到权重文件: {ckpt_path} (跳过)")
            continue

        # 提取名字，例如 "00000600_ema"
        ckpt_name = ckpt_path.stem 
        
        # 构建该权重的专属输出目录
        # 结果将保存为: /OUTPUT_ROOT/00000600_ema/metrics.json
        save_dir = os.path.join(OUTPUT_ROOT, ckpt_name)
        output_json = os.path.join(save_dir, "metrics.json")
        
        # 如果你想把图片也dump到这里，脚本里的 dump_path 会基于 output_json 的路径自动生成
        
        print(f"\n{'-'*60}")
        print(f"▶️ [{i+1}/{total_tasks}] 正在评估: {ckpt_name}")
        print(f"   📂 权重: {ckpt_path}")
        print(f"   📄 结果: {output_json}")
        
        # 检查是否已存在 (可选：如果想覆盖就注释掉这几行)
        if os.path.exists(output_json):
            print("   ⚠️ 结果文件已存在，跳过...")
            continue

        # 构建命令
        cmd = [
            "python", SCRIPT_REL_PATH,
            "--baseline", BASELINE_CODE,
            "--config", CONFIG_PATH,
            "--output", output_json,
            "--pretrained", str(ckpt_path),
            "--resolution_level", "9",
            "--dump_pred",   # 如果不想保存预测图片，注释掉这行
            "--ratio", str(SAMPLING_RATIO) # 传入采样率
        ]

        try:
            # 关键点：cwd=PROJECT_ROOT 确保 import 和相对路径正确
            subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
            print(f"✅ 完成: {ckpt_name}")
        
        except subprocess.CalledProcessError as e:
            print(f"❌ 评估失败 (Exit Code {e.returncode})")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")

    print(f"\n🎉 所有任务执行完毕！")

if __name__ == "__main__":
    run_batch_eval()