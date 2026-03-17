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

# 1. 权重根目录 (脚本会自动扫描这里面的 *_ema.pt 文件)
CHECKPOINT_ROOT = "/data1/szq/workspace/final-all/checkpoint"

# --- 🎯 权重筛选配置 (人性化读取规则) ---
EVAL_STEP_INTERVAL = 1200      # 【核心】每隔多少步评估一次？设为 100 表示只跑 0, 100, 200... (设为 None 或 0 则关闭)
EVAL_EXACT_STEPS = []       # 【白名单】额外强制评估的特定步数，无视间隔规则 (不需要则设为 [])
EVAL_MIN_STEP = 0             # 允许评估的最小步数
EVAL_MAX_STEP = 999999        # 允许评估的最大步数
IGNORE_LATEST = True          # 是否忽略 latest_ema.pt 文件？

# 2. 普通模型配置
SAMPLING_RATIO = 1.0
GPU_ID = "1"

# --- 任务 A: Bench & Oblique 配置 ---
TASK_A_ENABLE = True 
TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
TASK_A_OUTPUT_DIR = "/data1/szq/Table2-final-all2-EMA" 
TASK_A_DATASETS = [
    "/data1/szq/Val/Bench",
    "/data1/szq/Val/Oblique"
]

# --- 任务 B: Ground Datasets 配置 ---
TASK_B_ENABLE = True 
TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
TASK_B_OUTPUT_DIR = "/data1/szq/Table2-final-all2-EMA-Ground"
TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/baselines/moge.py" 
TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# --- 任务 C: 绘图配置 ---
PLOT_ENABLE = True    
PLOT_SAVE_PREFIX = "final_all2_ema" 

# ================= 💉 0时刻 Baseline 数据注入 =================

def get_baseline_data():
    """返回硬编码的 0 时刻评估结果，免去推理"""
    return [
        # ================= Bench (0 Step) =================
        {"Step": 0, "Source": "Bench", "Scene": "OVERALL", "AbsRel": 0.3702, "RMSE": 44.513, "a1.25": 0.253, "Type": "Summary"},
        {"Step": 0, "Source": "Bench", "Scene": "Campus", "AbsRel": 0.2933, "RMSE": 32.658, "a1.25": 0.257, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Factory", "AbsRel": 0.2196, "RMSE": 24.510, "a1.25": 0.499, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Farm", "AbsRel": 0.7076, "RMSE": 91.131, "a1.25": 0.000, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Gress", "AbsRel": 0.3114, "RMSE": 36.567, "a1.25": 0.254, "Type": "Detail"},
        
        # ================= Oblique (0 Step) =================
        {"Step": 0, "Source": "Oblique", "Scene": "OVERALL", "AbsRel": 0.4924, "RMSE": 71.841, "a1.25": 0.079, "Type": "Summary"},
        {"Step": 0, "Source": "Oblique", "Scene": "City", "AbsRel": 0.4827, "RMSE": 88.358, "a1.25": 0.051, "Type": "Detail"},
        {"Step": 0, "Source": "Oblique", "Scene": "Natural", "AbsRel": 0.4502, "RMSE": 43.121, "a1.25": 0.231, "Type": "Detail"},
        {"Step": 0, "Source": "Oblique", "Scene": "Rural", "AbsRel": 0.5666, "RMSE": 52.495, "a1.25": 0.003, "Type": "Detail"},

        # ================= Ground JSON (0 Step) =================
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

def get_sorted_checkpoints(root_dir):
    if not os.path.exists(root_dir):
        print(f"❌ 权重目录不存在: {root_dir}")
        return []
    
    # 针对普通模型，只匹配 *_ema.pt
    files = glob.glob(os.path.join(root_dir, "*_ema.pt"))
    selected_files = []
    
    def extract_step(path):
        stem = Path(path).stem
        if "latest" in stem:
            return -1  # 用 -1 代表 latest_ema
        match = re.search(r'(\d+)', stem)
        return int(match.group(1)) if match else -2 # 用 -2 代表解析失败的非标准文件

    for f in files:
        step = extract_step(f)
        
        # 1. 处理解析失败的文件
        if step == -2:
            continue
            
        # 2. 处理 latest_ema.pt
        if step == -1:
            if not IGNORE_LATEST:
                selected_files.append(f)
            continue
            
        is_selected = False
        
        # 3. 白名单特权：无视后面所有的限制
        if step in EVAL_EXACT_STEPS:
            is_selected = True
            
        # 4. 常规区间与间隔判定
        elif EVAL_MIN_STEP <= step <= EVAL_MAX_STEP:
            if EVAL_STEP_INTERVAL and EVAL_STEP_INTERVAL > 0:
                if step % EVAL_STEP_INTERVAL == 0:
                    is_selected = True
            else:
                is_selected = True # 没设间隔全要
                
        if is_selected:
            selected_files.append(f)

    # 去重并排序 (把 latest 放到最后排)
    selected_files = list(set(selected_files))
    selected_files = sorted(selected_files, key=lambda x: float('inf') if extract_step(x) == -1 else extract_step(x))
    
    return selected_files

def run_cmd(cmd_list, log_file, cwd=None, env=None):
    """执行命令，终端只提示简要信息，详细进度和报错写入 log_file"""
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
    os.makedirs(base_dir, exist_ok=True) # 确保权重子文件夹存在
    
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
                    "python", "a-infer.py",
                    "--input", dataset_path,
                    "--output", infer_out,
                    "--model", ckpt_path,  # 普通模型使用 --model
                    "--ratio", str(SAMPLING_RATIO),
                    "--resize", "0",
                    "--batch_size", "16"   # 引入 LoRA 脚本中的批处理优化
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
                if not file.endswith(".txt") and not file.endswith(".csv"): # 保留新增的详细csv报告
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
            # 🔥 普通模型使用 --pretrained，去掉了 --lora_config 等参数
            "--pretrained", ckpt_path, 
            "--resolution_level", "9",
            "--ratio", str(SAMPLING_RATIO)
        ]
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
    
    # 🔥 1. 注入 0 时刻硬编码的 Baseline 数据
    all_data.extend(get_baseline_data())

    if os.path.exists(TASK_A_OUTPUT_DIR):
        for subdir in os.listdir(TASK_A_OUTPUT_DIR):
            match = re.search(r'(\d+)', subdir)
            if match:
                step = int(match.group(1))
                if step == 0: continue # 0 用基准数据
                step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
                all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
                all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
    
    if os.path.exists(TASK_B_OUTPUT_DIR):
        for subdir in os.listdir(TASK_B_OUTPUT_DIR):
            match = re.search(r'(\d+)', subdir)
            if match:
                step = int(match.group(1))
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
    checkpoints = get_sorted_checkpoints(CHECKPOINT_ROOT)
    print(f"🔎 扫描到 {len(checkpoints)} 个 EMA 权重文件: {[Path(p).stem for p in checkpoints]}")
    
    if not checkpoints:
        print("未发现 *_ema.pt 文件，程序退出。")
        sys.exit(0)

    total_steps = len(checkpoints)
    for i, ckpt in enumerate(checkpoints):
        ckpt_name = Path(ckpt).stem
        
        # 提取数字步数判断是否为 0
        match = re.search(r'(\d+)', ckpt_name)
        step = int(match.group(1)) if match else -1
        
        print(f"\n{'='*80}")
        if step == 0:
            print(f"🔄 [Global Progress {i+1}/{total_steps}] Checkpoint: {ckpt_name} (跳过推理，使用预设 Baseline)")
            print(f"{'='*80}")
            if PLOT_ENABLE:
                print(f"📈 [Real-time Plotting] 绘制初始 Baseline 图表...")
                try: run_plotting()
                except Exception as e: print(f"⚠️ 画图失败: {e}")
            continue

        print(f"🔄 [Global Progress {i+1}/{total_steps}] 处理 Checkpoint: {ckpt_name}")
        print(f"{'='*80}")
        
        if TASK_A_ENABLE: process_single_ckpt_task_a(ckpt)
        if TASK_B_ENABLE: process_single_ckpt_task_b(ckpt)
        
        if PLOT_ENABLE:
            print(f"📈 [Real-time Plotting] 更新趋势图...")
            try: run_plotting()
            except Exception as e: print(f"⚠️ 画图暂时失败: {e}")

    print(f"\n🎉🎉🎉 全流程执行完毕！最终图表已生成。")
# import os
# import sys
# import subprocess
# import re
# import glob
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # ================= ⚙️ 全局配置区域 =================

# # 1. 权重根目录 (脚本会自动扫描这里面的 *_ema.pt 文件)
# CHECKPOINT_ROOT = "/home/szq/moge2/MoGe/workspace/final-head/checkpoint"

# # 2. 普通模型配置
# SAMPLING_RATIO = 0.2
# GPU_ID = "0"

# # --- 任务 A: Bench & Oblique 配置 ---
# TASK_A_ENABLE = True 
# TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
# TASK_A_OUTPUT_DIR = "/data1/szq/Table2-final-head-EMA" 
# TASK_A_DATASETS = [
#     "/data1/szq/Val/Bench",
#     "/data1/szq/Val/Oblique"
# ]

# # --- 任务 B: Ground Datasets 配置 ---
# TASK_B_ENABLE = True 
# TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
# TASK_B_OUTPUT_DIR = "/data1/szq/Table2-final-head-EMA-Ground"

# # 🔥 [已修复] 填入你刚刚找到的正确路径
# TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/baselines/moge.py" 

# TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
# TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# # --- 任务 C: 绘图配置 ---
# PLOT_ENABLE = True    
# PLOT_SAVE_PREFIX = "final_head_ema" 

# # ================= 🛠️ 辅助函数 =================

# def get_sorted_checkpoints(root_dir):
#     if not os.path.exists(root_dir):
#         print(f"❌ 权重目录不存在: {root_dir}")
#         return []
    
#     # 只匹配 *_ema.pt
#     files = glob.glob(os.path.join(root_dir, "*_ema.pt"))
    
#     def extract_step(path):
#         match = re.search(r'(\d+)', Path(path).stem)
#         return int(match.group(1)) if match else -1
    
#     files = sorted(files, key=extract_step)
#     return [f for f in files if extract_step(f) >= 0]

# def run_cmd(cmd_list, cwd=None, env=None):
#     cmd_str = " ".join(cmd_list)
#     print(f"⚡ Running: {cmd_str}")
#     subprocess.run(cmd_list, check=True, cwd=cwd, env=env)

# # ================= 🚀 模块 1: Task A (普通模型版) =================

# def process_single_ckpt_task_a(ckpt_path):
#     ckpt_name = Path(ckpt_path).stem 
#     print(f"\n   📍 [Task A] 处理权重: {ckpt_name}")
    
#     os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = GPU_ID
#     env["PYTHONPATH"] = f"{TASK_A_SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    
#     base_dir = os.path.join(TASK_A_OUTPUT_DIR, ckpt_name)
#     infer_out = os.path.join(base_dir, "Infer")
#     extract_out = os.path.join(base_dir, "Extracted")
    
#     report_bench = os.path.join(extract_out, "Bench", "Eval_Report_Bench.txt")
#     report_oblique = os.path.join(extract_out, "Oblique", "Eval_Report_Oblique_Pixel.txt")
    
#     if os.path.exists(report_bench) and os.path.exists(report_oblique):
#         print(f"      ✅ Task A 最终报告已存在，跳过。")
#         return

#     # 1. 推理
#     for dataset_path in TASK_A_DATASETS:
#         dataset_name = Path(dataset_path).name
#         expected_out = os.path.join(infer_out, dataset_name)
        
#         has_files = False
#         if os.path.exists(expected_out) and len(glob.glob(os.path.join(expected_out, "*.npy"))) > 0:
#             has_files = True
        
#         if has_files:
#             print(f"      ⏩ {dataset_name} 推理文件存在，跳过。")
#         else:
#             try:
#                 run_cmd([
#                     "python", "a-infer.py",
#                     "--input", dataset_path,
#                     "--output", infer_out,
#                     "--model", ckpt_path,  
#                     "--ratio", str(SAMPLING_RATIO),
#                     "--resize", "0"
#                 ], cwd=TASK_A_SCRIPT_DIR, env=env)
#             except Exception as e:
#                 print(f"      ❌ Task A 推理失败: {e}")

#     # 2. 抽取
#     need_extract = True
#     if os.path.exists(os.path.join(extract_out, "Bench")) and os.path.exists(os.path.join(extract_out, "Oblique")):
#          if len(os.listdir(os.path.join(extract_out, "Bench"))) > 0:
#              need_extract = False
    
#     if not need_extract:
#         print(f"      ⏩ 抽取文件存在，跳过。")
#     else:
#         try:
#             run_cmd([
#                 "python", "b-extra.py",
#                 "--input", infer_out,
#                 "--output", extract_out,
#                 "--target", "depth.npy",
#                 "--ext", ".npy"
#             ], cwd=TASK_A_SCRIPT_DIR, env=env)
#         except Exception as e:
#             print(f"      ❌ Task A 抽取失败: {e}")

#     # 3. 评估 Bench
#     if not os.path.exists(report_bench):
#         bench_pred = os.path.join(extract_out, "Bench")
#         if os.path.exists(bench_pred):
#             try:
#                 run_cmd([
#                     "python", "c-eval-bench.py",
#                     "--pred", bench_pred,
#                     "--gt", "/data1/szq/Val/Bench" 
#                 ], cwd=TASK_A_SCRIPT_DIR, env=env)
#             except: pass
    
#     # 4. 评估 Oblique
#     if not os.path.exists(report_oblique):
#         oblique_pred = os.path.join(extract_out, "Oblique")
#         if os.path.exists(oblique_pred):
#             try:
#                 run_cmd([
#                     "python", "c-eval-oblique.py",
#                     "--pred", oblique_pred,
#                     "--gt", "/data1/szq/Val/Oblique"
#                 ], cwd=TASK_A_SCRIPT_DIR, env=env)
#             except: pass

# # ================= 🚀 模块 2: Task B (普通模型版) =================

# def process_single_ckpt_task_b(ckpt_path):
#     ckpt_name = Path(ckpt_path).stem
#     print(f"\n   📍 [Task B] 处理权重: {ckpt_name}")

#     os.makedirs(TASK_B_OUTPUT_DIR, exist_ok=True)
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = GPU_ID
#     env["PYTHONPATH"] = f"{TASK_B_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
#     save_dir = os.path.join(TASK_B_OUTPUT_DIR, ckpt_name)
#     output_json = os.path.join(save_dir, "metrics.json")
    
#     if os.path.exists(output_json) and os.path.getsize(output_json) > 10:
#         print("      ✅ Task B metrics.json 已存在，跳过。")
#         return
        
#     try:
#         cmd = [
#             "python", TASK_B_EVAL_SCRIPT_REL,
#             "--baseline", TASK_B_BASELINE_SCRIPT,
#             "--config", TASK_B_DATASET_CONFIG,
#             "--output", output_json,
#             # 🔥 注意：普通模型通常用 --pretrained
#             # 如果报错 unrecognized arguments: --pretrained，请尝试改成 --load_weights
#             "--pretrained", ckpt_path, 
#             "--resolution_level", "9",
#             "--ratio", str(SAMPLING_RATIO)
#         ]
#         run_cmd(cmd, cwd=TASK_B_PROJECT_ROOT, env=env)
#     except Exception as e:
#         print(f"      ❌ Task B 评估失败: {e}")

# # ================= 🚀 模块 3: 汇总绘图 =================

# def clean_txt_scene_name(raw_name):
#     name = raw_name.strip()
#     if "Cleaned_Dataset_" in name: return name.replace("Cleaned_Dataset_", "")
#     if "[CAT]" in name: return name.replace("[CAT]", "").strip()
#     return name

# def parse_txt_file(file_path, source_type, step):
#     rows = []
#     if not os.path.exists(file_path): return rows
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
#     except: return rows
    
#     if source_type == "Bench": targets = ["OVERALL", "Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory", "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress"]
#     else: targets = ["OVERALL", "[CAT] City", "[CAT] Natural", "[CAT] Rural"]

#     for line in lines:
#         if "|" not in line or "---" in line: continue
#         parts = [p.strip() for p in line.split('|')]
#         if len(parts) < 6: continue
#         raw_name = parts[0]
#         if raw_name not in targets: continue
#         try:
#             rows.append({
#                 "Step": step,
#                 "Source": source_type,
#                 "Scene": clean_txt_scene_name(raw_name),
#                 "AbsRel": float(parts[2]),
#                 "RMSE": float(parts[3]),
#                 "a1.25": float(parts[5]),
#                 "Type": "Summary" if clean_txt_scene_name(raw_name).upper() == "OVERALL" else "Detail"
#             })
#         except: continue
#     return rows

# def parse_json_file(file_path, step):
#     rows = []
#     if not os.path.exists(file_path): return rows
#     try:
#         with open(file_path, 'r') as f: content = json.load(f)
#         for ds_name, metrics_group in content.items():
#             if "depth_metric" not in metrics_group: continue
#             metrics = metrics_group["depth_metric"]
#             if "rel" not in metrics: continue
#             is_summary = (ds_name.lower() == "mean")
#             rows.append({
#                 "Step": step,
#                 "Source": "Depth_JSON",
#                 "Scene": "Mean" if is_summary else ds_name,
#                 "AbsRel": float(metrics.get("rel", 0)),
#                 "RMSE": float(metrics.get("rmse", 0)),
#                 "a1.25": float(metrics.get("delta1", 0)),
#                 "Type": "Summary" if is_summary else "Detail"
#             })
#     except: pass
#     return rows

# def run_plotting():
#     print(f"\n{'='*80}")
#     print(f"📊 [Task C] 汇总绘图")
#     print(f"{'='*80}")
    
#     all_data = []
#     if os.path.exists(TASK_A_OUTPUT_DIR):
#         for subdir in os.listdir(TASK_A_OUTPUT_DIR):
#             match = re.search(r'(\d+)', subdir)
#             if match:
#                 step = int(match.group(1))
#                 step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
#                 all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
#                 all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
    
#     if os.path.exists(TASK_B_OUTPUT_DIR):
#         for subdir in os.listdir(TASK_B_OUTPUT_DIR):
#             match = re.search(r'(\d+)', subdir)
#             if match:
#                 step = int(match.group(1))
#                 all_data.extend(parse_json_file(os.path.join(TASK_B_OUTPUT_DIR, subdir, "metrics.json"), step))
                
#     if not all_data:
#         print("❌ 无数据，跳过绘图。")
#         return

#     df = pd.DataFrame(all_data).sort_values(by="Step")
#     sns.set_theme(style="whitegrid", font_scale=1.1)
    
#     def plot_metric(metric, filename, lower_better=True):
#         fig, axes = plt.subplots(1, 4, figsize=(28, 6))
        
#         # Summary
#         df_sum = df[df["Type"] == "Summary"].copy()
#         if not df_sum.empty:
#             df_sum["Legend"] = df_sum["Source"] + " " + df_sum["Scene"]
#             sns.lineplot(data=df_sum, x="Step", y=metric, hue="Legend", style="Legend", 
#                          markers=True, linewidth=3.5, markersize=10, ax=axes[0])
#             suffix = "↓" if lower_better else "↑"
#             axes[0].set_title(f"⭐ Global Summary ({metric} {suffix})", fontsize=16, fontweight='bold', color='darkred')
        
#         # Bench
#         df_bench = df[(df["Source"] == "Bench") & (df["Type"] == "Detail")]
#         if not df_bench.empty:
#             sns.lineplot(data=df_bench, x="Step", y=metric, hue="Scene", markers=True, ax=axes[1])
#             axes[1].set_title("Bench Sub-Scenes")

#         # Oblique
#         df_oblique = df[(df["Source"] == "Oblique") & (df["Type"] == "Detail")]
#         if not df_oblique.empty:
#             sns.lineplot(data=df_oblique, x="Step", y=metric, hue="Scene", markers=True, ax=axes[2])
#             axes[2].set_title("Oblique Categories")
            
#         # Ground JSON
#         df_json = df[(df["Source"] == "Depth_JSON") & (df["Type"] == "Detail")]
#         if not df_json.empty:
#             sns.lineplot(data=df_json, x="Step", y=metric, hue="Scene", style="Scene", markers=True, ax=axes[3])
#             axes[3].set_title("Other Datasets")
#             axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

#         for ax in axes:
#             ax.grid(True, linestyle='--', alpha=0.6)
#             ax.set_ylabel(metric)
#             ax.set_xlabel("Step")
        
#         # 图片保存到 Task A 输出目录
#         save_path = os.path.join(TASK_A_OUTPUT_DIR, filename)
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"🖼️ 已保存: {save_path}")

#     plot_metric("AbsRel", f"{PLOT_SAVE_PREFIX}_absrel.png", True)
#     plot_metric("RMSE", f"{PLOT_SAVE_PREFIX}_rmse.png", True)
#     plot_metric("a1.25", f"{PLOT_SAVE_PREFIX}_a125.png", False)

# # ================= 🚀 主程序入口 =================

# if __name__ == "__main__":
#     checkpoints = get_sorted_checkpoints(CHECKPOINT_ROOT)
#     print(f"🔎 扫描到 {len(checkpoints)} 个 EMA 权重文件")
    
#     if not checkpoints:
#         print("未发现 *_ema.pt 文件，程序退出。")
#         sys.exit(0)

#     total = len(checkpoints)
#     for i, ckpt in enumerate(checkpoints):
#         print(f"\n{'='*80}")
#         print(f"🔄 [Global Progress {i+1}/{total}] Checkpoint: {Path(ckpt).stem}")
#         print(f"{'='*80}")
        
#         if TASK_A_ENABLE: process_single_ckpt_task_a(ckpt)
#         if TASK_B_ENABLE: process_single_ckpt_task_b(ckpt)
        
#         if PLOT_ENABLE:
#             print(f"📈 [Real-time Plotting] 更新趋势图...")
#             try: run_plotting()
#             except Exception as e: print(f"⚠️ 画图失败: {e}")

#     print(f"\n🎉🎉🎉 全流程执行完毕！")


# # import os
# # import sys
# # import subprocess
# # import re
# # import glob
# # import json
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from pathlib import Path

# # # ================= ⚙️ 全局配置区域 =================

# # # 1. 权重根目录 (脚本会自动扫描这里面的 *_ema.pt 文件)
# # CHECKPOINT_ROOT = "/home/szq/moge2/MoGe/workspace/final-head/checkpoint"

# # # 2. 普通模型配置
# # SAMPLING_RATIO = 0.1
# # GPU_ID = "3"

# # # --- 任务 A: Bench & Oblique 配置 ---
# # TASK_A_ENABLE = True 
# # TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
# # # 🔥 结果输出目录 (建议区分 output 目录，以免和 LoRA 的混淆)
# # TASK_A_OUTPUT_DIR = "/data1/szq/Table2-final-head-EMA" 
# # TASK_A_DATASETS = [
# #     "/data1/szq/Val/Bench",
# #     "/data1/szq/Val/Oblique"
# # ]

# # # --- 任务 B: Ground Datasets 配置 ---
# # TASK_B_ENABLE = True 
# # TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
# # TASK_B_OUTPUT_DIR = "/data1/szq/Table2-final-head-EMA-Ground"
# # # 🔥 普通模型的接口脚本
# # TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/moge/scripts/baselines/moge.py" 
# # TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
# # TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# # # --- 任务 C: 绘图配置 ---
# # PLOT_ENABLE = True    
# # PLOT_SAVE_PREFIX = "final_head_ema" 

# # # ================= 🛠️ 辅助函数 =================

# # def get_sorted_checkpoints(root_dir):
# #     if not os.path.exists(root_dir):
# #         print(f"❌ 权重目录不存在: {root_dir}")
# #         return []
    
# #     # 🔥 [核心修改] 只匹配 *_ema.pt
# #     files = glob.glob(os.path.join(root_dir, "*_ema.pt"))
    
# #     def extract_step(path):
# #         # 提取文件名中的数字，例如 00000400_ema -> 400
# #         match = re.search(r'(\d+)', Path(path).stem)
# #         return int(match.group(1)) if match else -1
    
# #     files = sorted(files, key=extract_step)
# #     return [f for f in files if extract_step(f) >= 0]

# # def run_cmd(cmd_list, cwd=None, env=None):
# #     cmd_str = " ".join(cmd_list)
# #     print(f"⚡ Running: {cmd_str}")
# #     subprocess.run(cmd_list, check=True, cwd=cwd, env=env)

# # # ================= 🚀 模块 1: Task A (普通模型版) =================

# # def process_single_ckpt_task_a(ckpt_path):
# #     # ckpt_name 会是 "00000400_ema"，这样生成的文件夹名也是这个，不会和非ema混淆
# #     ckpt_name = Path(ckpt_path).stem 
# #     print(f"\n   📍 [Task A] 处理权重: {ckpt_name}")
    
# #     os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
    
# #     env = os.environ.copy()
# #     env["CUDA_VISIBLE_DEVICES"] = GPU_ID
# #     env["PYTHONPATH"] = f"{TASK_A_SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    
# #     base_dir = os.path.join(TASK_A_OUTPUT_DIR, ckpt_name)
# #     infer_out = os.path.join(base_dir, "Infer")
# #     extract_out = os.path.join(base_dir, "Extracted")
    
# #     report_bench = os.path.join(extract_out, "Bench", "Eval_Report_Bench.txt")
# #     report_oblique = os.path.join(extract_out, "Oblique", "Eval_Report_Oblique_Pixel.txt")
    
# #     if os.path.exists(report_bench) and os.path.exists(report_oblique):
# #         print(f"      ✅ Task A 最终报告已存在，跳过。")
# #         return

# #     # 1. 推理
# #     for dataset_path in TASK_A_DATASETS:
# #         dataset_name = Path(dataset_path).name
# #         expected_out = os.path.join(infer_out, dataset_name)
        
# #         has_files = False
# #         if os.path.exists(expected_out) and len(glob.glob(os.path.join(expected_out, "*.npy"))) > 0:
# #             has_files = True
        
# #         if has_files:
# #             print(f"      ⏩ {dataset_name} 推理文件存在，跳过。")
# #         else:
# #             try:
# #                 # 调用普通版 a-infer.py
# #                 run_cmd([
# #                     "python", "a-infer.py",
# #                     "--input", dataset_path,
# #                     "--output", infer_out,
# #                     "--model", ckpt_path,  
# #                     "--ratio", str(SAMPLING_RATIO),
# #                     "--resize", "0"
# #                 ], cwd=TASK_A_SCRIPT_DIR, env=env)
# #             except Exception as e:
# #                 print(f"      ❌ Task A 推理失败: {e}")

# #     # 2. 抽取
# #     need_extract = True
# #     if os.path.exists(os.path.join(extract_out, "Bench")) and os.path.exists(os.path.join(extract_out, "Oblique")):
# #          if len(os.listdir(os.path.join(extract_out, "Bench"))) > 0:
# #              need_extract = False
    
# #     if not need_extract:
# #         print(f"      ⏩ 抽取文件存在，跳过。")
# #     else:
# #         try:
# #             run_cmd([
# #                 "python", "b-extra.py",
# #                 "--input", infer_out,
# #                 "--output", extract_out,
# #                 "--target", "depth.npy",
# #                 "--ext", ".npy"
# #             ], cwd=TASK_A_SCRIPT_DIR, env=env)
# #         except Exception as e:
# #             print(f"      ❌ Task A 抽取失败: {e}")

# #     # 3. 评估 Bench
# #     if not os.path.exists(report_bench):
# #         bench_pred = os.path.join(extract_out, "Bench")
# #         if os.path.exists(bench_pred):
# #             try:
# #                 run_cmd([
# #                     "python", "c-eval-bench.py",
# #                     "--pred", bench_pred,
# #                     "--gt", "/data1/szq/Val/Bench" 
# #                 ], cwd=TASK_A_SCRIPT_DIR, env=env)
# #             except: pass
    
# #     # 4. 评估 Oblique
# #     if not os.path.exists(report_oblique):
# #         oblique_pred = os.path.join(extract_out, "Oblique")
# #         if os.path.exists(oblique_pred):
# #             try:
# #                 run_cmd([
# #                     "python", "c-eval-oblique.py",
# #                     "--pred", oblique_pred,
# #                     "--gt", "/data1/szq/Val/Oblique"
# #                 ], cwd=TASK_A_SCRIPT_DIR, env=env)
# #             except: pass

# # # ================= 🚀 模块 2: Task B (普通模型版) =================

# # def process_single_ckpt_task_b(ckpt_path):
# #     ckpt_name = Path(ckpt_path).stem
# #     print(f"\n   📍 [Task B] 处理权重: {ckpt_name}")

# #     os.makedirs(TASK_B_OUTPUT_DIR, exist_ok=True)
    
# #     env = os.environ.copy()
# #     env["CUDA_VISIBLE_DEVICES"] = GPU_ID
# #     env["PYTHONPATH"] = f"{TASK_B_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
# #     save_dir = os.path.join(TASK_B_OUTPUT_DIR, ckpt_name)
# #     output_json = os.path.join(save_dir, "metrics.json")
    
# #     if os.path.exists(output_json) and os.path.getsize(output_json) > 10:
# #         print("      ✅ Task B metrics.json 已存在，跳过。")
# #         return
        
# #     try:
# #         cmd = [
# #             "python", TASK_B_EVAL_SCRIPT_REL,
# #             "--baseline", TASK_B_BASELINE_SCRIPT,
# #             "--config", TASK_B_DATASET_CONFIG,
# #             "--output", output_json,
# #             # 普通模型传参
# #             "--pretrained", ckpt_path, 
# #             "--resolution_level", "9",
# #             "--ratio", str(SAMPLING_RATIO)
# #         ]
# #         run_cmd(cmd, cwd=TASK_B_PROJECT_ROOT, env=env)
# #     except Exception as e:
# #         print(f"      ❌ Task B 评估失败: {e}")

# # # ================= 🚀 模块 3: 汇总绘图 =================

# # def clean_txt_scene_name(raw_name):
# #     name = raw_name.strip()
# #     if "Cleaned_Dataset_" in name: return name.replace("Cleaned_Dataset_", "")
# #     if "[CAT]" in name: return name.replace("[CAT]", "").strip()
# #     return name

# # def parse_txt_file(file_path, source_type, step):
# #     rows = []
# #     if not os.path.exists(file_path): return rows
# #     try:
# #         with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
# #     except: return rows
    
# #     if source_type == "Bench": targets = ["OVERALL", "Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory", "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress"]
# #     else: targets = ["OVERALL", "[CAT] City", "[CAT] Natural", "[CAT] Rural"]

# #     for line in lines:
# #         if "|" not in line or "---" in line: continue
# #         parts = [p.strip() for p in line.split('|')]
# #         if len(parts) < 6: continue
# #         raw_name = parts[0]
# #         if raw_name not in targets: continue
# #         try:
# #             rows.append({
# #                 "Step": step,
# #                 "Source": source_type,
# #                 "Scene": clean_txt_scene_name(raw_name),
# #                 "AbsRel": float(parts[2]),
# #                 "RMSE": float(parts[3]),
# #                 "a1.25": float(parts[5]),
# #                 "Type": "Summary" if clean_txt_scene_name(raw_name).upper() == "OVERALL" else "Detail"
# #             })
# #         except: continue
# #     return rows

# # def parse_json_file(file_path, step):
# #     rows = []
# #     if not os.path.exists(file_path): return rows
# #     try:
# #         with open(file_path, 'r') as f: content = json.load(f)
# #         for ds_name, metrics_group in content.items():
# #             if "depth_metric" not in metrics_group: continue
# #             metrics = metrics_group["depth_metric"]
# #             if "rel" not in metrics: continue
# #             is_summary = (ds_name.lower() == "mean")
# #             rows.append({
# #                 "Step": step,
# #                 "Source": "Depth_JSON",
# #                 "Scene": "Mean" if is_summary else ds_name,
# #                 "AbsRel": float(metrics.get("rel", 0)),
# #                 "RMSE": float(metrics.get("rmse", 0)),
# #                 "a1.25": float(metrics.get("delta1", 0)),
# #                 "Type": "Summary" if is_summary else "Detail"
# #             })
# #     except: pass
# #     return rows

# # def run_plotting():
# #     print(f"\n{'='*80}")
# #     print(f"📊 [Task C] 汇总绘图")
# #     print(f"{'='*80}")
    
# #     all_data = []
# #     if os.path.exists(TASK_A_OUTPUT_DIR):
# #         for subdir in os.listdir(TASK_A_OUTPUT_DIR):
# #             # 这里可能会匹配到数字，确保能处理
# #             match = re.search(r'(\d+)', subdir)
# #             if match:
# #                 # 提取数字步数，因为文件夹名可能是 00000400_ema
# #                 step = int(match.group(1))
# #                 step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
# #                 all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
# #                 all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
    
# #     if os.path.exists(TASK_B_OUTPUT_DIR):
# #         for subdir in os.listdir(TASK_B_OUTPUT_DIR):
# #             match = re.search(r'(\d+)', subdir)
# #             if match:
# #                 step = int(match.group(1))
# #                 all_data.extend(parse_json_file(os.path.join(TASK_B_OUTPUT_DIR, subdir, "metrics.json"), step))
                
# #     if not all_data:
# #         print("❌ 无数据，跳过绘图。")
# #         return

# #     df = pd.DataFrame(all_data).sort_values(by="Step")
# #     sns.set_theme(style="whitegrid", font_scale=1.1)
    
# #     def plot_metric(metric, filename, lower_better=True):
# #         fig, axes = plt.subplots(1, 4, figsize=(28, 6))
        
# #         # Summary
# #         df_sum = df[df["Type"] == "Summary"].copy()
# #         if not df_sum.empty:
# #             df_sum["Legend"] = df_sum["Source"] + " " + df_sum["Scene"]
# #             sns.lineplot(data=df_sum, x="Step", y=metric, hue="Legend", style="Legend", 
# #                          markers=True, linewidth=3.5, markersize=10, ax=axes[0])
# #             suffix = "↓" if lower_better else "↑"
# #             axes[0].set_title(f"⭐ Global Summary ({metric} {suffix})", fontsize=16, fontweight='bold', color='darkred')
        
# #         # Bench
# #         df_bench = df[(df["Source"] == "Bench") & (df["Type"] == "Detail")]
# #         if not df_bench.empty:
# #             sns.lineplot(data=df_bench, x="Step", y=metric, hue="Scene", markers=True, ax=axes[1])
# #             axes[1].set_title("Bench Sub-Scenes")

# #         # Oblique
# #         df_oblique = df[(df["Source"] == "Oblique") & (df["Type"] == "Detail")]
# #         if not df_oblique.empty:
# #             sns.lineplot(data=df_oblique, x="Step", y=metric, hue="Scene", markers=True, ax=axes[2])
# #             axes[2].set_title("Oblique Categories")
            
# #         # Ground JSON
# #         df_json = df[(df["Source"] == "Depth_JSON") & (df["Type"] == "Detail")]
# #         if not df_json.empty:
# #             sns.lineplot(data=df_json, x="Step", y=metric, hue="Scene", style="Scene", markers=True, ax=axes[3])
# #             axes[3].set_title("Other Datasets")
# #             axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# #         for ax in axes:
# #             ax.grid(True, linestyle='--', alpha=0.6)
# #             ax.set_ylabel(metric)
# #             ax.set_xlabel("Step")
        
# #         # 图片保存到 Task A 输出目录
# #         save_path = os.path.join(TASK_A_OUTPUT_DIR, filename)
# #         plt.tight_layout()
# #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #         print(f"🖼️ 已保存: {save_path}")

# #     plot_metric("AbsRel", f"{PLOT_SAVE_PREFIX}_absrel.png", True)
# #     plot_metric("RMSE", f"{PLOT_SAVE_PREFIX}_rmse.png", True)
# #     plot_metric("a1.25", f"{PLOT_SAVE_PREFIX}_a125.png", False)

# # # ================= 🚀 主程序入口 =================

# # if __name__ == "__main__":
# #     checkpoints = get_sorted_checkpoints(CHECKPOINT_ROOT)
# #     print(f"🔎 扫描到 {len(checkpoints)} 个 EMA 权重文件")
    
# #     if not checkpoints:
# #         print("未发现 *_ema.pt 文件，程序退出。")
# #         sys.exit(0)

# #     total = len(checkpoints)
# #     for i, ckpt in enumerate(checkpoints):
# #         print(f"\n{'='*80}")
# #         print(f"🔄 [Global Progress {i+1}/{total}] Checkpoint: {Path(ckpt).stem}")
# #         print(f"{'='*80}")
        
# #         if TASK_A_ENABLE: process_single_ckpt_task_a(ckpt)
# #         if TASK_B_ENABLE: process_single_ckpt_task_b(ckpt)
        
# #         if PLOT_ENABLE:
# #             print(f"📈 [Real-time Plotting] 更新趋势图...")
# #             try: run_plotting()
# #             except Exception as e: print(f"⚠️ 画图失败: {e}")

# #     print(f"\n🎉🎉🎉 全流程执行完毕！")