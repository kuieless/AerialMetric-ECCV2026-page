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
import shutil
from datetime import datetime

# ================= ⚙️ 全局配置区域 =================

# 1. 🎯 直接指定你要跑的单个权重文件的绝对路径
SINGLE_CHECKPOINT_PATH = "/home/szq/moge2/MoGe/vitl-normal.pt"

# 2. 普通模型配置
SAMPLING_RATIO = 1.0
GPU_ID = "3"

# --- 任务 A: Bench & Oblique 配置 ---
TASK_A_ENABLE = True
TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
TASK_A_OUTPUT_DIR = "/data1/szq/Table2-final-baseline-with-intrin2" 
TASK_A_DATASETS = [
    "/data1/szq/Val/Bench-ori",
    "/data1/szq/Val/Oblique"
]

# --- 任务 B: Ground Datasets 配置 ---
TASK_B_ENABLE = False 
TASK_B_USE_ORACLE = True
TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
TASK_B_OUTPUT_DIR = "/data1/szq/Table2-final-neck48-3-Ground-norm"
TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/baselines/moge.py" 
TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# --- 任务 C: 绘图配置 ---
PLOT_ENABLE = True    
PLOT_SAVE_PREFIX = "final_neck48-3" 

# ================= 💉 0时刻 Baseline 数据注入 =================

def get_baseline_data():
    """返回硬编码的 0 时刻评估结果，免去推理"""
    return [
        # Bench
        {"Step": 0, "Source": "Bench", "Scene": "OVERALL", "AbsRel": 0.3702, "RMSE": 44.513, "a1.25": 0.253, "Type": "Summary"},
        {"Step": 0, "Source": "Bench", "Scene": "Campus", "AbsRel": 0.2933, "RMSE": 32.658, "a1.25": 0.257, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Factory", "AbsRel": 0.2196, "RMSE": 24.510, "a1.25": 0.499, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Farm", "AbsRel": 0.7076, "RMSE": 91.131, "a1.25": 0.000, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Gress", "AbsRel": 0.3114, "RMSE": 36.567, "a1.25": 0.254, "Type": "Detail"},
        
        # Oblique
        {"Step": 0, "Source": "Oblique", "Scene": "OVERALL", "AbsRel": 0.4924, "RMSE": 71.841, "a1.25": 0.079, "Type": "Summary"},
        {"Step": 0, "Source": "Oblique", "Scene": "City", "AbsRel": 0.4827, "RMSE": 88.358, "a1.25": 0.051, "Type": "Detail"},
        {"Step": 0, "Source": "Oblique", "Scene": "Natural", "AbsRel": 0.4502, "RMSE": 43.121, "a1.25": 0.231, "Type": "Detail"},
        {"Step": 0, "Source": "Oblique", "Scene": "Rural", "AbsRel": 0.5666, "RMSE": 52.495, "a1.25": 0.003, "Type": "Detail"},

        # Ground JSON
        {"Step": 0, "Source": "Depth_JSON", "Scene": "NYUv2", "AbsRel": 0.0689, "RMSE": 0.23, "a1.25": 0.967, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "KITTI", "AbsRel": 0.1750, "RMSE": 4.26, "a1.25": 0.647, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "ETH3D", "AbsRel": 0.0998, "RMSE": 0.74, "a1.25": 0.888, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "iBims", "AbsRel": 0.1450, "RMSE": 0.54, "a1.25": 0.804, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "DDAD", "AbsRel": 0.1470, "RMSE": 5.21, "a1.25": 0.751, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "DIODE", "AbsRel": 0.1660, "RMSE": 2.48, "a1.25": 0.679, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "HAMMER", "AbsRel": 0.2600, "RMSE": 0.2, "a1.25": 0.652, "Type": "Detail"},
        {"Step": 0, "Source": "Depth_JSON", "Scene": "Mean", "AbsRel": 0.152, "RMSE": 1.95, "a1.25": 0.770, "Type": "Summary"},
    ]

# ================= 🛠️ 辅助函数 =================

def run_cmd(cmd_list, log_file, cwd=None, env=None):
    cmd_str = " ".join(cmd_list)
    script_name = cmd_list[1] if len(cmd_list) > 1 else cmd_list[0]
    
    print(f"      ▶️ 执行: {script_name} ... (详细输出见日志: {os.path.basename(log_file)})")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚡ Running: {cmd_str}\n")
        subprocess.run(cmd_list, check=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)

# ================= 🚀 模块 1: Task A (普通模型版) =================

def process_single_ckpt_task_a(ckpt_path):
    ckpt_name = Path(ckpt_path).stem 
    print(f"\n   📍 [Task A] 处理权重: {ckpt_name}")
    
    os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONPATH"] = f"{TASK_A_SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    
    base_dir = os.path.join(TASK_A_OUTPUT_DIR, ckpt_name)
    os.makedirs(base_dir, exist_ok=True)
    
    infer_out = os.path.join(base_dir, "Infer")
    extract_out = os.path.join(base_dir, "Extracted")
    run_log = os.path.join(base_dir, "subprocess_run.log")
    
    report_bench = os.path.join(extract_out, "Bench", "Eval_Report_Bench.txt")
    report_oblique = os.path.join(extract_out, "Oblique", "Eval_Report_Oblique_Pixel.txt")
    
    if os.path.exists(report_bench) and os.path.exists(report_oblique):
        print(f"      ✅ Task A 最终报告已存在，跳过。")
        return

    # 1. 推理
    for dataset_path in TASK_A_DATASETS:
        dataset_name = Path(dataset_path).name
        expected_out = os.path.join(infer_out, dataset_name)
        
        has_files = False
        if os.path.exists(expected_out) and len(glob.glob(os.path.join(expected_out, "*.npy"))) > 0:
            has_files = True
        
        if has_files:
            print(f"      ⏩ {dataset_name} 推理文件存在，跳过。")
        else:
            try:
                run_cmd([
                    "python", "a-infer_norm.py",
                    "--input", dataset_path,
                    "--output", infer_out,
                    "--model", ckpt_path,  # 普通模型使用 --model
                    "--ratio", str(SAMPLING_RATIO),
                    "--resize", "0",
                    "--batch_size", "1" 
                ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except Exception as e:
                print(f"      ❌ Task A 推理失败: {e}")

    # 2. 抽取
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
            ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
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
                ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
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
                ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass

    # ================= 🧹 5. 磁盘清理逻辑 =================
    if os.path.exists(report_bench) and os.path.exists(report_oblique):
        print(f"      🧹 评估完成，开始清理冗余文件以节省磁盘空间...")
        if os.path.exists(infer_out):
            shutil.rmtree(infer_out, ignore_errors=True)
            
        for root_dir, dirs, files in os.walk(extract_out):
            for file in files:
                if not file.endswith(".txt") and not file.endswith(".csv"): 
                    try: os.remove(os.path.join(root_dir, file))
                    except: pass
        print(f"      ✨ 清理完毕！仅保留 txt/csv 报告与日志。")

# ================= 🚀 模块 2: Task B (普通模型版) =================

def process_single_ckpt_task_b(ckpt_path):
    ckpt_name = Path(ckpt_path).stem
    print(f"\n   📍 [Task B] 处理权重: {ckpt_name}")

    os.makedirs(TASK_B_OUTPUT_DIR, exist_ok=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONPATH"] = f"{TASK_B_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    save_dir = os.path.join(TASK_B_OUTPUT_DIR, ckpt_name)
    os.makedirs(save_dir, exist_ok=True)
    
    run_log = os.path.join(save_dir, "task_b_run.log")
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
            "--pretrained", ckpt_path, 
            "--resolution_level", "9",
            "--ratio", str(SAMPLING_RATIO)
        ]
        if TASK_B_USE_ORACLE:
                    cmd.append("--oracle")
        run_cmd(cmd, log_file=run_log, cwd=TASK_B_PROJECT_ROOT, env=env)
    except Exception as e:
        print(f"      ❌ Task B 评估失败: {e}")

# ================= 🚀 模块 3: 汇总绘图 =================

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
    print(f"📊 [Task C] 汇总绘图 (包含注入的 0 时刻 Baseline)")
    print(f"{'='*80}")
    
    all_data = []
    
    # 注入 0 时刻硬编码的 Baseline 数据
    all_data.extend(get_baseline_data())

    if os.path.exists(TASK_A_OUTPUT_DIR):
        for subdir in os.listdir(TASK_A_OUTPUT_DIR):
            match = re.search(r'(\d+)', subdir)
            # 如果文件名里找不到数字（比如叫 baseline_model），为了能画进图表里，默认给它安排步数 1
            step = int(match.group(1)) if match else 1
            if step == 0: continue # 0 用基准数据
            step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
            all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
            all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
    
    if os.path.exists(TASK_B_OUTPUT_DIR):
        for subdir in os.listdir(TASK_B_OUTPUT_DIR):
            match = re.search(r'(\d+)', subdir)
            step = int(match.group(1)) if match else 1
            if step == 0: continue
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
            axes[0].set_title(f"Global Summary ({metric} {suffix})", fontsize=16, fontweight='bold', color='darkred')
        
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
            # 强制 x 轴显示整数
            ax.xaxis.get_major_locator().set_params(integer=True)
        
        # 图片保存到 Task A 输出目录
        os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(TASK_A_OUTPUT_DIR, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"🖼️ 已保存: {save_path}")

    plot_metric("AbsRel", f"{PLOT_SAVE_PREFIX}_absrel.png", True)
    plot_metric("RMSE", f"{PLOT_SAVE_PREFIX}_rmse.png", True)
    plot_metric("a1.25", f"{PLOT_SAVE_PREFIX}_a125.png", False)

# ================= 🚀 主程序入口 =================

if __name__ == "__main__":
    if not os.path.exists(SINGLE_CHECKPOINT_PATH):
        print(f"❌ 找不到权重文件: {SINGLE_CHECKPOINT_PATH}")
        sys.exit(1)

    print(f"🔎 准备运行指定的权重: {SINGLE_CHECKPOINT_PATH}")
    ckpt_name = Path(SINGLE_CHECKPOINT_PATH).stem
    
    print(f"\n{'='*80}")
    print(f"🔄 处理 Checkpoint: {ckpt_name}")
    print(f"{'='*80}")
    
    if TASK_A_ENABLE: process_single_ckpt_task_a(SINGLE_CHECKPOINT_PATH)
    if TASK_B_ENABLE: process_single_ckpt_task_b(SINGLE_CHECKPOINT_PATH)
    
    if PLOT_ENABLE:
        print(f"📈 [Real-time Plotting] 正在生成评估对比图...")
        try: run_plotting()
        except Exception as e: print(f"⚠️ 画图失败: {e}")

    print(f"\n🎉🎉🎉 全流程执行完毕！最终评估报告与图表已生成。")