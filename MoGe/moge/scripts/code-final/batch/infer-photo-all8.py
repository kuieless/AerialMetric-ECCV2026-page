import os
import sys
import subprocess
import re
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================= ⚙️ 全局配置区域 =================

# 1. 权重根目录
CHECKPOINT_ROOT = "/home/szq/moge2/MoGe/workspace/lora-batch8-16-lessground4-with-UE/checkpoint"

# 2. LoRA 通用配置
# LORA_CONFIG_PATH = "/home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json"
LORA_CONFIG_PATH = "/home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UE.json"

SAMPLING_RATIO = 0.12
GPU_ID = "2"

# --- 任务 A: Bench & Oblique 配置 ---
TASK_A_ENABLE = True 
TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
TASK_A_OUTPUT_DIR = "/data1/szq/Infer-Final/Table2-1/Table2-head-LoRA-8-groundless4-UE"
TASK_A_DATASETS = [
    "/data1/szq/Val/Bench",
    "/data1/szq/Val/Oblique"
]

# --- 任务 B: Ground Datasets 配置 ---
TASK_B_ENABLE = True 
TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
TASK_B_OUTPUT_DIR = "/data1/szq/Table2-head-LoRA-8-Ground-groundless4-UE"
TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/moge/scripts/eval_baseline_lora8.py"
TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# --- 任务 C: 绘图配置 ---
PLOT_ENABLE = True    
PLOT_SAVE_PREFIX = "final_lora8-groundless4-UE" 

# ================= 🛠️ 辅助函数 =================

def get_sorted_checkpoints(root_dir):
    if not os.path.exists(root_dir):
        print(f"❌ 权重目录不存在: {root_dir}")
        return []
    files = glob.glob(os.path.join(root_dir, "*.pt"))
    def extract_step(path):
        match = re.search(r'(\d+)', Path(path).stem)
        return int(match.group(1)) if match else -1
    files = sorted(files, key=extract_step)
    return [f for f in files if extract_step(f) >= 0]

def run_cmd(cmd_list, cwd=None, env=None):
    cmd_str = " ".join(cmd_list)
    print(f"⚡ Running: {cmd_str}")
    subprocess.run(cmd_list, check=True, cwd=cwd, env=env)

# ================= 🚀 模块 1: 单个权重的 Task A 逻辑 =================

def process_single_ckpt_task_a(ckpt_path):
    ckpt_name = Path(ckpt_path).stem
    print(f"\n   📍 [Task A] 处理权重: {ckpt_name}")
    
    os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
    
    # 设置 Task A 专属环境
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONPATH"] = f"{TASK_A_SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    
    # 定义路径
    base_dir = os.path.join(TASK_A_OUTPUT_DIR, ckpt_name)
    infer_out = os.path.join(base_dir, "Infer")
    extract_out = os.path.join(base_dir, "Extracted")
    
    report_bench = os.path.join(extract_out, "Bench", "Eval_Report_Bench.txt")
    report_oblique = os.path.join(extract_out, "Oblique", "Eval_Report_Oblique_Pixel.txt")
    
    # 检查完成状态
    if os.path.exists(report_bench) and os.path.exists(report_oblique):
        print(f"      ✅ Task A 最终报告已存在，跳过。")
        return

    # 1. 推理 (Inference)
    for dataset_path in TASK_A_DATASETS:
        dataset_name = Path(dataset_path).name
        expected_out = os.path.join(infer_out, dataset_name)
        
        # 简单检查非空
        has_files = False
        if os.path.exists(expected_out) and len(glob.glob(os.path.join(expected_out, "*.npy"))) > 0:
            has_files = True
        
        if has_files:
            print(f"      ⏩ {dataset_name} 推理文件存在，跳过。")
        else:
            try:
                run_cmd([
                    "python", "a-infer_lora8.py",
                    "--input", dataset_path,
                    "--output", infer_out,
                    "--config", LORA_CONFIG_PATH,
                    "--weight", ckpt_path,
                    "--ratio", str(SAMPLING_RATIO),
                    "--resize", "0",
                    "--batch_size", "8"  # <--- 🚀 加上这个参数，8或者16，看你的显存能不能撑住
                ], cwd=TASK_A_SCRIPT_DIR, env=env)
            except Exception as e:
                print(f"      ❌ Task A 推理失败: {e}")

    # 2. 抽取 (Extraction)
    need_extract = True
    if os.path.exists(os.path.join(extract_out, "Bench")) and os.path.exists(os.path.join(extract_out, "Oblique")):
         if len(os.listdir(os.path.join(extract_out, "Bench"))) > 0:
             need_extract = False
    
    if not need_extract:
        print(f"      ⏩ 抽取文件存在，跳过。")
    else:
        try:
            run_cmd([
                "python", "b-extra.py",
                "--input", infer_out,
                "--output", extract_out,
                "--target", "depth.npy",
                "--ext", ".npy"
            ], cwd=TASK_A_SCRIPT_DIR, env=env)
        except Exception as e:
            print(f"      ❌ Task A 抽取失败: {e}")

    # 3. 评估 Bench
    if not os.path.exists(report_bench):
        bench_pred = os.path.join(extract_out, "Bench")
        if os.path.exists(bench_pred):
            try:
                run_cmd([
                    "python", "c-eval-bench.py",
                    "--pred", bench_pred,
                    "--gt", "/data1/szq/Val/Bench" 
                ], cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass
    
    # 4. 评估 Oblique
    if not os.path.exists(report_oblique):
        oblique_pred = os.path.join(extract_out, "Oblique")
        if os.path.exists(oblique_pred):
            try:
                run_cmd([
                    "python", "c-eval-oblique.py",
                    "--pred", oblique_pred,
                    "--gt", "/data1/szq/Val/Oblique"
                ], cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass

# ================= 🚀 模块 2: 单个权重的 Task B 逻辑 =================

def process_single_ckpt_task_b(ckpt_path):
    ckpt_name = Path(ckpt_path).stem
    print(f"\n   📍 [Task B] 处理权重: {ckpt_name}")

    os.makedirs(TASK_B_OUTPUT_DIR, exist_ok=True)
    
    # 设置 Task B 专属环境
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONPATH"] = f"{TASK_B_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    save_dir = os.path.join(TASK_B_OUTPUT_DIR, ckpt_name)
    output_json = os.path.join(save_dir, "metrics.json")
    
    if os.path.exists(output_json) and os.path.getsize(output_json) > 10:
        print("      ✅ Task B metrics.json 已存在，跳过。")
        return
        
    try:
        cmd = [
            "python", TASK_B_EVAL_SCRIPT_REL,
            "--baseline", TASK_B_BASELINE_SCRIPT,
            "--config", TASK_B_DATASET_CONFIG,
            "--output", output_json,
            "--lora_config", LORA_CONFIG_PATH,
            "--lora_weight", ckpt_path,
            "--resolution_level", "9",
            "--ratio", str(SAMPLING_RATIO)
        ]
        run_cmd(cmd, cwd=TASK_B_PROJECT_ROOT, env=env)
    except Exception as e:
        print(f"      ❌ Task B 评估失败: {e}")

# ================= 🚀 模块 3: 数据汇总与绘图 (保持全局) =================

def clean_txt_scene_name(raw_name):
    name = raw_name.strip()
    if "Cleaned_Dataset_" in name: return name.replace("Cleaned_Dataset_", "")
    if "[CAT]" in name: return name.replace("[CAT]", "").strip()
    return name

def parse_txt_file(file_path, source_type, step):
    rows = []
    if not os.path.exists(file_path): return rows
    try:
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    except: return rows

    if source_type == "Bench": targets = ["OVERALL", "Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory", "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress"]
    else: targets = ["OVERALL", "[CAT] City", "[CAT] Natural", "[CAT] Rural"]

    for line in lines:
        if "|" not in line or "---" in line: continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        raw_name = parts[0]
        if raw_name not in targets: continue
        try:
            rows.append({
                "Step": step,
                "Source": source_type,
                "Scene": clean_txt_scene_name(raw_name),
                "AbsRel": float(parts[2]),
                "RMSE": float(parts[3]),
                "a1.25": float(parts[5]),
                "Type": "Summary" if clean_txt_scene_name(raw_name).upper() == "OVERALL" else "Detail"
            })
        except: continue
    return rows

def parse_json_file(file_path, step):
    rows = []
    if not os.path.exists(file_path): return rows
    try:
        with open(file_path, 'r') as f: content = json.load(f)
        for ds_name, metrics_group in content.items():
            if "depth_metric" not in metrics_group: continue
            metrics = metrics_group["depth_metric"]
            if "rel" not in metrics: continue
            is_summary = (ds_name.lower() == "mean")
            rows.append({
                "Step": step,
                "Source": "Depth_JSON",
                "Scene": "Mean" if is_summary else ds_name,
                "AbsRel": float(metrics.get("rel", 0)),
                "RMSE": float(metrics.get("rmse", 0)),
                "a1.25": float(metrics.get("delta1", 0)),
                "Type": "Summary" if is_summary else "Detail"
            })
    except: pass
    return rows

def run_plotting():
    print(f"\n{'='*80}")
    print(f"📊 [Task C] 汇总绘图")
    print(f"{'='*80}")
    
    all_data = []
    if os.path.exists(TASK_A_OUTPUT_DIR):
        for subdir in os.listdir(TASK_A_OUTPUT_DIR):
            if subdir.isdigit():
                step = int(subdir)
                step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
                all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
                all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
    
    if os.path.exists(TASK_B_OUTPUT_DIR):
        for subdir in os.listdir(TASK_B_OUTPUT_DIR):
            if subdir.isdigit():
                step = int(subdir)
                all_data.extend(parse_json_file(os.path.join(TASK_B_OUTPUT_DIR, subdir, "metrics.json"), step))
                
    if not all_data:
        print("❌ 无数据，跳过绘图。")
        return

    df = pd.DataFrame(all_data).sort_values(by="Step")
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    def plot_metric(metric, filename, lower_better=True):
        fig, axes = plt.subplots(1, 4, figsize=(28, 6))
        
        # Summary
        df_sum = df[df["Type"] == "Summary"].copy()
        if not df_sum.empty:
            df_sum["Legend"] = df_sum["Source"] + " " + df_sum["Scene"]
            sns.lineplot(data=df_sum, x="Step", y=metric, hue="Legend", style="Legend", 
                         markers=True, linewidth=3.5, markersize=10, ax=axes[0])
            suffix = "↓" if lower_better else "↑"
            axes[0].set_title(f"⭐ Global Summary ({metric} {suffix})", fontsize=16, fontweight='bold', color='darkred')
        
        # Bench
        df_bench = df[(df["Source"] == "Bench") & (df["Type"] == "Detail")]
        if not df_bench.empty:
            sns.lineplot(data=df_bench, x="Step", y=metric, hue="Scene", markers=True, ax=axes[1])
            axes[1].set_title("Bench Sub-Scenes")

        # Oblique
        df_oblique = df[(df["Source"] == "Oblique") & (df["Type"] == "Detail")]
        if not df_oblique.empty:
            sns.lineplot(data=df_oblique, x="Step", y=metric, hue="Scene", markers=True, ax=axes[2])
            axes[2].set_title("Oblique Categories")
            
        # Ground JSON
        df_json = df[(df["Source"] == "Depth_JSON") & (df["Type"] == "Detail")]
        if not df_json.empty:
            sns.lineplot(data=df_json, x="Step", y=metric, hue="Scene", style="Scene", markers=True, ax=axes[3])
            axes[3].set_title("Other Datasets")
            axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylabel(metric)
            ax.set_xlabel("Step")
            
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"🖼️ 已保存: {filename}")

    plot_metric("AbsRel", f"{PLOT_SAVE_PREFIX}_absrel.png", True)
    plot_metric("RMSE", f"{PLOT_SAVE_PREFIX}_rmse.png", True)
    plot_metric("a1.25", f"{PLOT_SAVE_PREFIX}_a125.png", False)

# ================= 🚀 主程序入口 =================

# if __name__ == "__main__":
    
#     checkpoints = get_sorted_checkpoints(CHECKPOINT_ROOT)
#     print(f"🔎 扫描到 {len(checkpoints)} 个权重文件: {[Path(p).stem for p in checkpoints]}")
    
#     if not checkpoints:
#         print("未发现 .pt 文件，程序退出。")
#         sys.exit(0)

#     # 🔥 [核心逻辑变更] 遍历 Checkpoint，内部依次执行 A 和 B
#     total_steps = len(checkpoints)
#     for i, ckpt in enumerate(checkpoints):
#         ckpt_name = Path(ckpt).stem
#         print(f"\n{'='*80}")
#         print(f"🔄 [Global Progress {i+1}/{total_steps}] 处理 Checkpoint: {ckpt_name}")
#         print(f"{'='*80}")
        
#         # 1. 跑 Task A (如果开启)
#         if TASK_A_ENABLE:
#             process_single_ckpt_task_a(ckpt)
            
#         # 2. 跑 Task B (如果开启)
#         if TASK_B_ENABLE:
#             process_single_ckpt_task_b(ckpt)
            
#     # 3. 所有跑完后，统一画图
#     if PLOT_ENABLE:
#         run_plotting()
        
#     print(f"\n🎉🎉🎉 全流程执行完毕！")
if __name__ == "__main__":
    # 1. 扫描权重
    checkpoints = get_sorted_checkpoints(CHECKPOINT_ROOT)
    print(f"🔎 扫描到 {len(checkpoints)} 个权重文件: {[Path(p).stem for p in checkpoints]}")
    
    if not checkpoints:
        print("未发现 .pt 文件，程序退出。")
        sys.exit(0)

    # 2. 逐个处理
    total_steps = len(checkpoints)
    for i, ckpt in enumerate(checkpoints):
        ckpt_name = Path(ckpt).stem
        print(f"\n{'='*80}")
        print(f"🔄 [Global Progress {i+1}/{total_steps}] 处理 Checkpoint: {ckpt_name}")
        print(f"{'='*80}")
        
        # --- Step 1: 跑 Task A (Bench & Oblique) ---
        if TASK_A_ENABLE:
            process_single_ckpt_task_a(ckpt)
            
        # --- Step 2: 跑 Task B (Ground Datasets) ---
        if TASK_B_ENABLE:
            process_single_ckpt_task_b(ckpt)
            
        # --- Step 3: 🔥 立即画图 (实时更新) ---
        # 每次跑完一个权重，立刻重新扫描所有已完成的数据并画图
        if PLOT_ENABLE:
            print(f"📈 [Real-time Plotting] 更新趋势图...")
            try:
                run_plotting()
            except Exception as e:
                print(f"⚠️ 画图暂时失败 (可能是数据还不完整): {e}")

    print(f"\n🎉🎉🎉 全流程执行完毕！最终图表已生成。")