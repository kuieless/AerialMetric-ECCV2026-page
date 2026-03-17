import os
import sys
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ================= ⚙️ 用户配置区域 =================

# 1. 想要扫描的 LoRA 权重列表 (这里填 LoRA 的 .pt 文件路径)
CHECKPOINTS = [






# "/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000100.pt",
# "/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000200.pt",
# "/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000300.pt",
"/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000400.pt",
"/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000500.pt",
"/home/szq/moge2/MoGe/workspace/lora-batch16-32/checkpoint/00000600.pt",
]

# 2. LoRA 训练配置文件路径 (所有权重共用一个 Config)
# 🔥 [新增] 必须指定这个 JSON 文件
LORA_CONFIG_PATH = "/home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json"

# 3. 基础配置
# 请确保你的 a-infer-lora.py 放在这个目录下
SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final" 
OUTPUT_BASE = "/data1/szq/Infer-Final/Table2/Table2-head-LoRA-16-32"  # 修改一下输出目录名，避免和 Base 混淆

# 4. 目标数据集
TARGET_DATASETS = [
    "/data1/szq/Val/Bench",
    "/data1/szq/Val/Oblique"
]

# 5. 推理采样率
SAMPLING_RATIO = 0.2  

# 6. 图表保存路径
PLOT_SAVE_PATH = os.path.join(OUTPUT_BASE, "training_progress_lora.png")

# ================= 🛠️ 核心功能函数 (保持不变) =================

def run_cmd(cmd_list, log_file=None, cwd=None):
    cmd_list = [str(x) for x in cmd_list]
    cmd_str = " ".join(cmd_list)
    if cwd:
        print(f"⚡ Running (in {cwd}): {cmd_str}")
    else:
        print(f"⚡ Running: {cmd_str}")
    
    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=cwd) 
    else:
        subprocess.run(cmd_list, check=True, cwd=cwd)

def parse_txt_report(report_path):
    metrics = {}
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "OVERALL" in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) > 3:
                    metrics['AbsRel'] = float(parts[2])
                    metrics['RMSE'] = float(parts[3])
                break
    except Exception as e:
        print(f"⚠️ 解析 TXT 报告出错 ({report_path}): {e}")
    return metrics

# ================= 🚀 主循环 =================

def main():
    results = [] 
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    print(f"🚀 开始 LoRA 批量扫描，采样率: {SAMPLING_RATIO * 100}%")
    print(f"📂 目标数据集: {TARGET_DATASETS}")
    print(f"📄 LoRA Config: {LORA_CONFIG_PATH}")

    for ckpt_path in CHECKPOINTS:
        # 1. 提取 Checkpoint 的名字和步数
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"❌ 权重不存在: {ckpt_path}, 跳过...")
            continue

        ckpt_name = ckpt_path.stem 
        try:
            step = int(re.search(r'\d+', ckpt_name).group())
        except:
            step = 0
        
        print(f"\n{'='*60}")
        print(f"▶️ 处理 LoRA Checkpoint: {ckpt_name} (Step: {step})")
        print(f"{'='*60}")

        # 定义输出目录
        infer_out = os.path.join(OUTPUT_BASE, ckpt_name, "Infer")
        extract_out = os.path.join(OUTPUT_BASE, ckpt_name, "Extracted")
        
        # --- Step 1: 推理 (a-infer-lora.py) ---
        # 🔥 [修改重点] 参数改为适配 LoRA 脚本
        for dataset_path in TARGET_DATASETS:
            dataset_name = Path(dataset_path).name 
            
            expected_output_dir = os.path.join(infer_out, dataset_name)
            
            if not os.path.exists(expected_output_dir):
                print(f"running inference for {dataset_name}...")
                
                # 假设你的 LoRA 推理脚本叫 a-infer-lora.py
                run_cmd([
                    "python", os.path.join(SCRIPT_DIR, "a-infer_lora.py"),
                    "--input", dataset_path,      # 输入目录
                    "--output", infer_out,        # 输出根目录
                    "--config", LORA_CONFIG_PATH, # 🔥 新增: LoRA 配置文件
                    "--weight", str(ckpt_path),   # 🔥 修改: LoRA 权重文件 (原先是 --model)
                    "--ratio", SAMPLING_RATIO,
                    "--resize", "0"               # 0 代表 None
                ])
            else:
                print(f"⏩ {dataset_name} 推理结果已存在，跳过...")

        # --- Step 2: 抽取 (b-extra.py) ---
        # (保持不变)
        if not os.path.exists(extract_out):
            if os.path.exists(infer_out):
                run_cmd([
                    "python", os.path.join(SCRIPT_DIR, "b-extra.py"),
                    "--input", infer_out,    
                    "--output", extract_out,
                    "--target", "depth.npy",
                    "--ext", ".npy"
                ])
            else:
                print(f"⚠️ 警告: 推理目录 {infer_out} 不存在，无法进行抽取。")
        else:
            print("⏩ 抽取结果已存在，跳过...")

        # --- Step 3: 评估 Bench & Oblique (保持不变) ---
        current_result = {'step': step, 'name': ckpt_name}
        
        # 3.1 Bench 评估
        bench_pred = os.path.join(extract_out, "Bench")
        # 这里假设你的 GT 还在原位置，如果变了请修改
        bench_gt = os.path.join("/data1/szq/Val", "Bench") 
        
        if os.path.exists(bench_pred):
            print("Running Bench Eval...")
            run_cmd([
                "python", os.path.join(SCRIPT_DIR, "c-eval-bench.py"),
                "--pred", bench_pred,
                "--gt", bench_gt
            ])
            b_metrics = parse_txt_report(os.path.join(bench_pred, "Eval_Report_Bench.txt"))
            if b_metrics:
                current_result['Bench_AbsRel'] = b_metrics.get('AbsRel')
                current_result['Bench_RMSE'] = b_metrics.get('RMSE')
        
        # 3.2 Oblique 评估
        oblique_pred = os.path.join(extract_out, "Oblique")
        oblique_gt = os.path.join("/data1/szq/Val", "Oblique")
        
        if os.path.exists(oblique_pred):
            print("Running Oblique Eval...")
            run_cmd([
                "python", os.path.join(SCRIPT_DIR, "c-eval-oblique.py"),
                "--pred", oblique_pred,
                "--gt", oblique_gt
            ])
            o_metrics = parse_txt_report(os.path.join(oblique_pred, "Eval_Report_Oblique_Pixel.txt"))
            if o_metrics:
                current_result['Oblique_AbsRel'] = o_metrics.get('AbsRel')
                current_result['Oblique_RMSE'] = o_metrics.get('RMSE')

        results.append(current_result)
        print(f"📊 当前 Step {step} 汇总: {current_result}")

    # ================= 📈 绘图 (保持不变) =================
    
    if not results:
        print("❌ 没有产生任何结果，请检查路径。")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by='step')
    
    csv_path = os.path.join(OUTPUT_BASE, "bench_oblique_results_lora.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 所有数据已保存至: {csv_path}")
    
    plt.figure(figsize=(10, 6))
    has_plot = False
    if 'Bench_AbsRel' in df.columns and not df['Bench_AbsRel'].isnull().all():
        plt.plot(df['step'], df['Bench_AbsRel'], marker='o', label='Bench AbsRel')
        has_plot = True
    if 'Oblique_AbsRel' in df.columns and not df['Oblique_AbsRel'].isnull().all():
        plt.plot(df['step'], df['Oblique_AbsRel'], marker='s', label='Oblique AbsRel')
        has_plot = True

    if has_plot:
        plt.title(f"LoRA Validation Metrics (Sampling Ratio: {SAMPLING_RATIO})")
        plt.xlabel("Checkpoint Step")
        plt.ylabel("AbsRel (Lower is Better)")
        plt.grid(True)
        plt.legend()
        plt.savefig(PLOT_SAVE_PATH)
        print(f"📈 趋势图已保存至: {PLOT_SAVE_PATH}")
    
    if 'Bench_AbsRel' in df.columns and not df['Bench_AbsRel'].isnull().all():
        best_idx = df['Bench_AbsRel'].idxmin()
        best_row = df.loc[best_idx]
        print(f"\n🏆 最佳 Checkpoint (Bench): Step {best_row['step']} (AbsRel: {best_row['Bench_AbsRel']:.4f})")

if __name__ == "__main__":
    main()