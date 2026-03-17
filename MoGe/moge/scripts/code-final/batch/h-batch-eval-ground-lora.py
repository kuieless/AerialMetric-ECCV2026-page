import os
import sys
import subprocess
from pathlib import Path

# ================= ⚙️ 配置区域 =================

# 1. 想要评估的 LoRA 权重列表
CHECKPOINTS = [
# "/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000100.pt",
# "/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000200.pt",
# "/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000300.pt",

"/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000400.pt",
"/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000500.pt",
"/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000600.pt",
]

# 2. LoRA 训练配置文件 (必须指定，用于加载 Base Model)
LORA_CONFIG_PATH = "/home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json"

# 3. 输出根目录
OUTPUT_ROOT = "/data1/szq/Table2-head-LoRA-16-Ground"

# 4. 指定 GPU
GPU_ID = "1"

# 5. 采样率 (0.2 = 20%)
SAMPLING_RATIO = 0.2

# 6. 项目路径配置
PROJECT_ROOT = "/home/szq/moge2/MoGe"
SCRIPT_REL_PATH = "moge/scripts/eval_baseline.py"  # 评估框架脚本 (不变)

# 🔥 [修改重点] 指向你刚刚保存的 LoRA 接口文件
BASELINE_CODE = "/home/szq/moge2/MoGe/moge/scripts/eval_baseline_lora.py" 
CONFIG_PATH = "configs/eval/all_benchmarks.json"

# ================= 🚀 执行逻辑 =================

def run_batch_eval():
    # 确保输出根目录存在
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    total_tasks = len(CHECKPOINTS)
    print(f"🔥 准备开始 LoRA 批量评估，共 {total_tasks} 个模型...")
    print(f"📍 项目根目录: {PROJECT_ROOT}")
    print(f"💾 输出目录: {OUTPUT_ROOT}")
    print(f"📄 LoRA Config: {LORA_CONFIG_PATH}")
    print(f"📄 Baseline Interface: {BASELINE_CODE}")
    print(f"⚙️  GPU: {GPU_ID} | Ratio: {SAMPLING_RATIO}")

    for i, ckpt_path in enumerate(CHECKPOINTS):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"❌ 找不到权重文件: {ckpt_path} (跳过)")
            continue

        ckpt_name = ckpt_path.stem 
        
        # 构建该权重的专属输出目录
        save_dir = os.path.join(OUTPUT_ROOT, ckpt_name)
        output_json = os.path.join(save_dir, "metrics.json")
        
        print(f"\n{'-'*60}")
        print(f"▶️ [{i+1}/{total_tasks}] 正在评估: {ckpt_name}")
        print(f"   📂 权重: {ckpt_path}")
        print(f"   📄 结果: {output_json}")
        
        if os.path.exists(output_json):
            print("   ⚠️ 结果文件已存在，跳过...")
            continue

        # 构建命令
        # 注意：这里参数发生了变化，适配 LoRA 接口
        cmd = [
            "python", SCRIPT_REL_PATH,        # 运行 eval_baseline.py
            "--baseline", BASELINE_CODE,      # 加载 baselines/moge_lora.py
            "--config", CONFIG_PATH,          # 数据集配置
            "--output", output_json,          # 结果输出路径
            
            # 🔥 LoRA 特有参数 (传递给 moge_lora.py 的 load 函数)
            "--lora_config", LORA_CONFIG_PATH, 
            "--lora_weight", str(ckpt_path),
            
            # 通用参数
            "--resolution_level", "9",
            # "--dump_pred",                  # 是否保存图片
            "--ratio", str(SAMPLING_RATIO)    # 采样率
        ]

        try:
            # 关键点：cwd=PROJECT_ROOT 确保 import 正确
            subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
            print(f"✅ 完成: {ckpt_name}")
        
        except subprocess.CalledProcessError as e:
            print(f"❌ 评估失败 (Exit Code {e.returncode})")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")

    print(f"\n🎉 所有 LoRA 评估任务执行完毕！")

if __name__ == "__main__":
    run_batch_eval()