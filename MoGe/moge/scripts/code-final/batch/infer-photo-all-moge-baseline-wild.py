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
# import shutil
# from datetime import datetime

# # ================= ⚙️ 全局配置区域 =================

# # 1. 🎯 直接指定你要跑的单个权重文件的绝对路径
# SINGLE_CHECKPOINT_PATH = "/data1/szq/workspace/final-neck-lossfine/checkpoint/00004800_ema.pt"

# # 2. 普通模型配置
# SAMPLING_RATIO = 1.0
# GPU_ID = "0"

# # --- 任务 A: Bench & Oblique 配置 ---
# TASK_A_ENABLE = True 
# TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
# TASK_A_OUTPUT_DIR = "/data1/szq/Table2-final-neck48-3" 
# TASK_A_DATASETS = [
#     "/data1/szq/Val/Bench",
#     "/data1/szq/Val/Oblique"
# ]

# # --- 任务 B: Ground Datasets 配置 ---
# TASK_B_ENABLE = True 
# TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
# TASK_B_OUTPUT_DIR = "/data1/szq/Table2-final-neck48-3-Ground"
# TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/baselines/moge.py" 
# TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
# TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# # --- 任务 C: 绘图配置 ---
# PLOT_ENABLE = True    
# PLOT_SAVE_PREFIX = "final_neck48-3" 

# # ================= 💉 0时刻 Baseline 数据注入 =================

# def get_baseline_data():
#     """返回硬编码的 0 时刻评估结果，免去推理"""
#     return [
#         # Bench
#         {"Step": 0, "Source": "Bench", "Scene": "OVERALL", "AbsRel": 0.3702, "RMSE": 44.513, "a1.25": 0.253, "Type": "Summary"},
#         {"Step": 0, "Source": "Bench", "Scene": "Campus", "AbsRel": 0.2933, "RMSE": 32.658, "a1.25": 0.257, "Type": "Detail"},
#         {"Step": 0, "Source": "Bench", "Scene": "Factory", "AbsRel": 0.2196, "RMSE": 24.510, "a1.25": 0.499, "Type": "Detail"},
#         {"Step": 0, "Source": "Bench", "Scene": "Farm", "AbsRel": 0.7076, "RMSE": 91.131, "a1.25": 0.000, "Type": "Detail"},
#         {"Step": 0, "Source": "Bench", "Scene": "Gress", "AbsRel": 0.3114, "RMSE": 36.567, "a1.25": 0.254, "Type": "Detail"},
        
#         # Oblique
#         {"Step": 0, "Source": "Oblique", "Scene": "OVERALL", "AbsRel": 0.4924, "RMSE": 71.841, "a1.25": 0.079, "Type": "Summary"},
#         {"Step": 0, "Source": "Oblique", "Scene": "City", "AbsRel": 0.4827, "RMSE": 88.358, "a1.25": 0.051, "Type": "Detail"},
#         {"Step": 0, "Source": "Oblique", "Scene": "Natural", "AbsRel": 0.4502, "RMSE": 43.121, "a1.25": 0.231, "Type": "Detail"},
#         {"Step": 0, "Source": "Oblique", "Scene": "Rural", "AbsRel": 0.5666, "RMSE": 52.495, "a1.25": 0.003, "Type": "Detail"},

#         # Ground JSON
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "NYUv2", "AbsRel": 0.0689, "RMSE": 0.23, "a1.25": 0.967, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "KITTI", "AbsRel": 0.1750, "RMSE": 4.26, "a1.25": 0.647, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "ETH3D", "AbsRel": 0.0998, "RMSE": 0.74, "a1.25": 0.888, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "iBims", "AbsRel": 0.1450, "RMSE": 0.54, "a1.25": 0.804, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "DDAD", "AbsRel": 0.1470, "RMSE": 5.21, "a1.25": 0.751, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "DIODE", "AbsRel": 0.1660, "RMSE": 2.48, "a1.25": 0.679, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "HAMMER", "AbsRel": 0.2600, "RMSE": 0.2, "a1.25": 0.652, "Type": "Detail"},
#         {"Step": 0, "Source": "Depth_JSON", "Scene": "Mean", "AbsRel": 0.152, "RMSE": 1.95, "a1.25": 0.770, "Type": "Summary"},
#     ]

# # ================= 🛠️ 辅助函数 =================

# def run_cmd(cmd_list, log_file, cwd=None, env=None):
#     cmd_str = " ".join(cmd_list)
#     script_name = cmd_list[1] if len(cmd_list) > 1 else cmd_list[0]
    
#     print(f"      ▶️ 执行: {script_name} ... (详细输出见日志: {os.path.basename(log_file)})")
    
#     with open(log_file, "a", encoding="utf-8") as f:
#         f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚡ Running: {cmd_str}\n")
#         subprocess.run(cmd_list, check=True, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)

# # ================= 🚀 模块 1: Task A (普通模型版) =================

# def process_single_ckpt_task_a(ckpt_path):
#     ckpt_name = Path(ckpt_path).stem 
#     print(f"\n   📍 [Task A] 处理权重: {ckpt_name}")
    
#     os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = GPU_ID
#     env["PYTHONPATH"] = f"{TASK_A_SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
    
#     base_dir = os.path.join(TASK_A_OUTPUT_DIR, ckpt_name)
#     os.makedirs(base_dir, exist_ok=True)
    
#     infer_out = os.path.join(base_dir, "Infer")
#     extract_out = os.path.join(base_dir, "Extracted")
#     run_log = os.path.join(base_dir, "subprocess_run.log")
    
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
#                     "--model", ckpt_path,  # 普通模型使用 --model
#                     "--ratio", str(SAMPLING_RATIO),
#                     "--resize", "0",
#                     "--batch_size", "8" 
#                 ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
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
#             ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
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
#                 ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
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
#                 ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
#             except: pass

#     # ================= 🧹 5. 磁盘清理逻辑 =================
#     if os.path.exists(report_bench) and os.path.exists(report_oblique):
#         print(f"      🧹 评估完成，开始清理冗余文件以节省磁盘空间...")
#         if os.path.exists(infer_out):
#             shutil.rmtree(infer_out, ignore_errors=True)
            
#         for root_dir, dirs, files in os.walk(extract_out):
#             for file in files:
#                 if not file.endswith(".txt") and not file.endswith(".csv"): 
#                     try: os.remove(os.path.join(root_dir, file))
#                     except: pass
#         print(f"      ✨ 清理完毕！仅保留 txt/csv 报告与日志。")

# # ================= 🚀 模块 2: Task B (普通模型版) =================

# def process_single_ckpt_task_b(ckpt_path):
#     ckpt_name = Path(ckpt_path).stem
#     print(f"\n   📍 [Task B] 处理权重: {ckpt_name}")

#     os.makedirs(TASK_B_OUTPUT_DIR, exist_ok=True)
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = GPU_ID
#     env["PYTHONPATH"] = f"{TASK_B_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
#     save_dir = os.path.join(TASK_B_OUTPUT_DIR, ckpt_name)
#     os.makedirs(save_dir, exist_ok=True)
    
#     run_log = os.path.join(save_dir, "task_b_run.log")
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
#             "--pretrained", ckpt_path, 
#             "--resolution_level", "9",
#             "--ratio", str(SAMPLING_RATIO)
#         ]
#         run_cmd(cmd, log_file=run_log, cwd=TASK_B_PROJECT_ROOT, env=env)
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
#     print(f"📊 [Task C] 汇总绘图 (包含注入的 0 时刻 Baseline)")
#     print(f"{'='*80}")
    
#     all_data = []
    
#     # 注入 0 时刻硬编码的 Baseline 数据
#     all_data.extend(get_baseline_data())

#     if os.path.exists(TASK_A_OUTPUT_DIR):
#         for subdir in os.listdir(TASK_A_OUTPUT_DIR):
#             match = re.search(r'(\d+)', subdir)
#             # 如果文件名里找不到数字（比如叫 baseline_model），为了能画进图表里，默认给它安排步数 1
#             step = int(match.group(1)) if match else 1
#             if step == 0: continue # 0 用基准数据
#             step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
#             all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
#             all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
    
#     if os.path.exists(TASK_B_OUTPUT_DIR):
#         for subdir in os.listdir(TASK_B_OUTPUT_DIR):
#             match = re.search(r'(\d+)', subdir)
#             step = int(match.group(1)) if match else 1
#             if step == 0: continue
#             all_data.extend(parse_json_file(os.path.join(TASK_B_OUTPUT_DIR, subdir, "metrics.json"), step))
                
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
#             axes[0].set_title(f"Global Summary ({metric} {suffix})", fontsize=16, fontweight='bold', color='darkred')
        
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
#             # 强制 x 轴显示整数
#             ax.xaxis.get_major_locator().set_params(integer=True)
        
#         # 图片保存到 Task A 输出目录
#         os.makedirs(TASK_A_OUTPUT_DIR, exist_ok=True)
#         save_path = os.path.join(TASK_A_OUTPUT_DIR, filename)
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         print(f"🖼️ 已保存: {save_path}")

#     plot_metric("AbsRel", f"{PLOT_SAVE_PREFIX}_absrel.png", True)
#     plot_metric("RMSE", f"{PLOT_SAVE_PREFIX}_rmse.png", True)
#     plot_metric("a1.25", f"{PLOT_SAVE_PREFIX}_a125.png", False)

# # ================= 🚀 主程序入口 =================

# if __name__ == "__main__":
#     if not os.path.exists(SINGLE_CHECKPOINT_PATH):
#         print(f"❌ 找不到权重文件: {SINGLE_CHECKPOINT_PATH}")
#         sys.exit(1)

#     print(f"🔎 准备运行指定的权重: {SINGLE_CHECKPOINT_PATH}")
#     ckpt_name = Path(SINGLE_CHECKPOINT_PATH).stem
    
#     print(f"\n{'='*80}")
#     print(f"🔄 处理 Checkpoint: {ckpt_name}")
#     print(f"{'='*80}")
    
#     if TASK_A_ENABLE: process_single_ckpt_task_a(SINGLE_CHECKPOINT_PATH)
#     if TASK_B_ENABLE: process_single_ckpt_task_b(SINGLE_CHECKPOINT_PATH)
    
#     if PLOT_ENABLE:
#         print(f"📈 [Real-time Plotting] 正在生成评估对比图...")
#         try: run_plotting()
#         except Exception as e: print(f"⚠️ 画图失败: {e}")

#     print(f"\n🎉🎉🎉 全流程执行完毕！最终评估报告与图表已生成。")

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
GPU_ID = "0"

# 🌟 新增：Wild 数据集黑名单过滤 (直接复用你跑出来的列表)
EXCLUDE_WILD_SCENES = ['TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_318', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_877', 'scene_014', 'scene_035', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_024', 'THE_AMAZING_CHICHESTER_CATHEDRAL___Clip_Pack_292___Chichester_City_Centre_Drone_Stock_Footage_25_2oGluwGopaE_scene_000', 'scene_018', 'scene_047', 'Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_018', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_430', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_087', 'scene_048', 'Orbit_Shot_Drone_o9b3kPi3t3w_scene_000', 'Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_010', 'Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_032', 'Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_007', 'Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_001', 'scene_109', 'scene_033', 'Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_008', 'scene_074', 'Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_008', 'Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_012', 'scene_105', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_917', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_265', 'scene_056', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_836', 'Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_011', 'scene_045', 'Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_021', 'scene_114', 'Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_009', 'West_Bromwich_Albion_Football_Stadium_Stock_Footage__38kZf4qyKQ_scene_000', 'scene_061', 'scene_103', 'scene_075', 'scene_053', 'scene_108', 'scene_067', 'Orbit_Shot_of_The_Roman_Colosseum_4K_Free_Download_Amazing_Arial_View_Nlw5AfgKpQg_scene_000', 'videoplayback2_scene_000', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_050', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_179', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_545', 'Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_013', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_084', 'scene_102', 'TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_415']

# ================= 🎯 数据集自主控制开关 =================
ENABLE_BENCH   = False
ENABLE_OBLIQUE = False
ENABLE_WILD    = True     
ENABLE_GROUND  = False     

# --- 任务 A: Bench & Oblique & Wild 配置 ---
TASK_A_SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
TASK_A_OUTPUT_DIR = "/data1/szq/Table2-final-wild3" 

DATASET_PATHS = {
    "Bench": "/data1/szq/Val/Bench",
    "Oblique": "/data1/szq/Val/Oblique",
    "Wild": "/data1/szq/Val/Wild"
}

# --- 任务 B: Ground Datasets 配置 ---
TASK_B_PROJECT_ROOT = "/home/szq/moge2/MoGe"
TASK_B_OUTPUT_DIR = "/data1/szq/Table2-final-wild3"
TASK_B_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/baselines/moge.py" 
TASK_B_EVAL_SCRIPT_REL = "moge/scripts/eval_baseline.py"
TASK_B_DATASET_CONFIG = "configs/eval/all_benchmarks.json"

# --- 任务 C: 绘图配置 ---
PLOT_ENABLE = True    
PLOT_SAVE_PREFIX = "final_wild2" 

# ================= 💉 0时刻 Baseline 数据注入 =================

def get_baseline_data():
    return [
        {"Step": 0, "Source": "Bench", "Scene": "OVERALL", "AbsRel": 0.3702, "RMSE": 44.513, "a1.25": 0.253, "Type": "Summary"},
        {"Step": 0, "Source": "Bench", "Scene": "Campus", "AbsRel": 0.2933, "RMSE": 32.658, "a1.25": 0.257, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Factory", "AbsRel": 0.2196, "RMSE": 24.510, "a1.25": 0.499, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Farm", "AbsRel": 0.7076, "RMSE": 91.131, "a1.25": 0.000, "Type": "Detail"},
        {"Step": 0, "Source": "Bench", "Scene": "Gress", "AbsRel": 0.3114, "RMSE": 36.567, "a1.25": 0.254, "Type": "Detail"},
        
        {"Step": 0, "Source": "Oblique", "Scene": "OVERALL", "AbsRel": 0.4924, "RMSE": 71.841, "a1.25": 0.079, "Type": "Summary"},
        {"Step": 0, "Source": "Oblique", "Scene": "City", "AbsRel": 0.4827, "RMSE": 88.358, "a1.25": 0.051, "Type": "Detail"},
        {"Step": 0, "Source": "Oblique", "Scene": "Natural", "AbsRel": 0.4502, "RMSE": 43.121, "a1.25": 0.231, "Type": "Detail"},
        {"Step": 0, "Source": "Oblique", "Scene": "Rural", "AbsRel": 0.5666, "RMSE": 52.495, "a1.25": 0.003, "Type": "Detail"},

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
    
    active_datasets = {}
    if ENABLE_BENCH:   active_datasets["Bench"] = DATASET_PATHS["Bench"]
    if ENABLE_OBLIQUE: active_datasets["Oblique"] = DATASET_PATHS["Oblique"]
    if ENABLE_WILD:    active_datasets["Wild"] = DATASET_PATHS["Wild"]

    if not active_datasets:
        print("      ⚠️ 无需运行 Task A，所有相关数据集开关均已关闭。")
        return

    # 1. 推理 (Inference)
    for ds_name, ds_path in active_datasets.items():
        actual_input_path = ds_path
        temp_link_dir = None
        
        # 💡 黑名单过滤魔法：为 Wild 数据集动态构建“干净”的输入目录 (普通模型同样适用)
        if ds_name == "Wild" and EXCLUDE_WILD_SCENES:
            temp_link_dir = os.path.join(infer_out, ".Wild_Filtered_Input")
            os.makedirs(temp_link_dir, exist_ok=True)
            print(f"      🛡️ 启动过滤模式：排除 {len(EXCLUDE_WILD_SCENES)} 个异常场景...")
            
            for scene in os.listdir(ds_path):
                if scene not in EXCLUDE_WILD_SCENES and os.path.isdir(os.path.join(ds_path, scene)):
                    src_img_dir = os.path.join(ds_path, scene, "image")
                    
                    if os.path.exists(src_img_dir):
                        # 创建真实的场景和 image 文件夹
                        dst_img_dir = os.path.join(temp_link_dir, scene, "image")
                        os.makedirs(dst_img_dir, exist_ok=True)
                        
                        # 遍历源 image 文件夹，给里面的图片建软链接
                        for img_file in os.listdir(src_img_dir):
                            src_img = os.path.join(src_img_dir, img_file)
                            dst_img = os.path.join(dst_img_dir, img_file)
                            if not os.path.exists(dst_img) and os.path.isfile(src_img):
                                os.symlink(src_img, dst_img)
                                
            actual_input_path = temp_link_dir # 偷梁换柱

        expected_out = os.path.join(infer_out, ds_name)
        has_files = os.path.exists(expected_out) and len(list(Path(expected_out).rglob("*.npy"))) > 0
        
        if has_files:
            print(f"      ⏩ {ds_name} 推理文件存在，跳过。")
        else:
            try:
                # 🚀 注意这里：严格保留了普通模型专属参数 (a-infer.py 和 --model)
                run_cmd([
                    "python", "a-infer.py",
                    "--input", actual_input_path,
                    "--output", expected_out,
                    "--model", ckpt_path,  
                    "--ratio", str(SAMPLING_RATIO),
                    "--resize", "0",
                    "--batch_size", "8" 
                ], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except Exception as e:
                print(f"      ❌ Task A ({ds_name}) 推理失败: {e}")
            finally:
                if temp_link_dir and os.path.exists(temp_link_dir):
                    shutil.rmtree(temp_link_dir, ignore_errors=True)

    # 2. 抽取 (Extraction)
    if not os.path.exists(extract_out) or len(os.listdir(extract_out)) == 0:
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

    # 3. 评估 (Evaluation) 
    if ENABLE_BENCH:
        bench_pred = os.path.join(extract_out, "Bench")
        report_bench = os.path.join(bench_pred, "Eval_Report_Bench.txt")
        if not os.path.exists(report_bench) and os.path.exists(bench_pred):
            try: run_cmd(["python", "c-eval-bench.py", "--pred", bench_pred, "--gt", DATASET_PATHS["Bench"]], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass
    
    if ENABLE_OBLIQUE:
        oblique_pred = os.path.join(extract_out, "Oblique")
        report_oblique = os.path.join(oblique_pred, "Eval_Report_Oblique_Pixel.txt")
        if not os.path.exists(report_oblique) and os.path.exists(oblique_pred):
            try: run_cmd(["python", "c-eval-oblique.py", "--pred", oblique_pred, "--gt", DATASET_PATHS["Oblique"]], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass

    # 🌟 新增：评估 Wild 数据集 (双轨评估)
    if ENABLE_WILD:
        wild_pred = os.path.join(extract_out, "Wild")
        
        # 多量程深度评估
        report_wild_multi = os.path.join(wild_pred, "Eval_Report_Wild_MultiRange.txt")
        if not os.path.exists(report_wild_multi) and os.path.exists(wild_pred):
            try: run_cmd(["python", "c-eval-wild.py", "--pred", wild_pred, "--gt", DATASET_PATHS["Wild"]], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass
            
        # FoV 与深度联合评估
        report_wild_fov = os.path.join(wild_pred, "fov_analysis_details.csv")
        if not os.path.exists(report_wild_fov) and os.path.exists(wild_pred):
            try: run_cmd(["python", "c-eval-wild-fov.py", "--pred", wild_pred, "--gt", DATASET_PATHS["Wild"]], log_file=run_log, cwd=TASK_A_SCRIPT_DIR, env=env)
            except: pass

    # ================= 🧹 5. 磁盘清理逻辑 =================
# ================= 🧹 5. 磁盘清理逻辑 =================
# ================= 🧹 5. 磁盘清理逻辑 (🌟保留原始深度张量版) =================
    print(f"      🧹 阶段完成，开始清理...")
    
    # ⚠️ 已经移除了对 infer_out 目录的清理循环，完美保留所有的原始浮点深度数据！
                    
    # 智能清理 Extracted 目录：保护报告、图像以及深度数据 (.npy)
    if os.path.exists(extract_out):
        for root_dir, dirs, files in os.walk(extract_out):
            for file in files:
                # 🌟 白名单里加上了 .npy，凡是不在白名单里的临时文件统统删掉
                if not file.endswith((".txt", ".csv", ".png", ".jpg", ".npy")): 
                    try: os.remove(os.path.join(root_dir, file))
                    except: pass
                    
    print(f"      ✨ 清理完毕！已成功保留原始深度数据(.npy)、可视化结果图与评估报告。")

# ================= 🚀 模块 2: Task B (普通模型版) =================

def process_single_ckpt_task_b(ckpt_path):
    ckpt_name = Path(ckpt_path).stem
    print(f"\n   📍 [Task B (Ground)] 处理权重: {ckpt_name}")

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
    
    targets = ["OVERALL"]
    if source_type == "Bench": targets += ["Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory", "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress"]
    elif source_type == "Oblique": targets += ["[CAT] City", "[CAT] Natural", "[CAT] Rural"]

    for line in lines:
        if "|" not in line or "---" in line: continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        raw_name = parts[0]
        if raw_name not in targets: continue
        try:
            rows.append({
                "Step": step, "Source": source_type, "Scene": clean_txt_scene_name(raw_name),
                "AbsRel": float(parts[2]), "RMSE": float(parts[3]), "a1.25": float(parts[5]),
                "Type": "Summary" if clean_txt_scene_name(raw_name).upper() == "OVERALL" else "Detail"
            })
        except: continue
    return rows

def parse_wild_txt_file(file_path, step):
    rows = []
    if not os.path.exists(file_path): return rows
    try:
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    except: return rows

    current_range = None
    for line in lines:
        match = re.search(r'0 - (\d+)m', line)
        if match: 
            current_range = match.group(1)
            continue
        
        if not current_range: continue
        if "|" not in line or "---" in line or "Scene" in line: continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        raw_name = parts[0]
        
        try:
            rows.append({
                "Step": step, 
                "Source": f"Wild_{current_range}m",
                "Scene": clean_txt_scene_name(raw_name),
                "AbsRel": float(parts[2]),
                "RMSE": float(parts[3]),
                "a1.25": float(parts[5]),
                "Type": "Summary" if raw_name.upper() == "OVERALL" else "Detail"
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
                "Step": step, "Source": "Depth_JSON", "Scene": "Mean" if is_summary else ds_name,
                "AbsRel": float(metrics.get("rel", 0)), "RMSE": float(metrics.get("rmse", 0)),
                "a1.25": float(metrics.get("delta1", 0)), "Type": "Summary" if is_summary else "Detail"
            })
    except: pass
    return rows

def run_plotting():
    print(f"\n{'='*80}")
    print(f"📊 [Task C] 汇总绘图 (包含注入的 0 时刻 Baseline)")
    print(f"{'='*80}")
    
    all_data = []
    all_data.extend(get_baseline_data())

    if os.path.exists(TASK_A_OUTPUT_DIR):
        for subdir in os.listdir(TASK_A_OUTPUT_DIR):
            match = re.search(r'(\d+)', subdir)
            step = int(match.group(1)) if match else 1
            if step == 0: continue 
            step_path = os.path.join(TASK_A_OUTPUT_DIR, subdir)
            if ENABLE_BENCH:   all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Bench", "Eval_Report_Bench.txt"), "Bench", step))
            if ENABLE_OBLIQUE: all_data.extend(parse_txt_file(os.path.join(step_path, "Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt"), "Oblique", step))
            if ENABLE_WILD:    all_data.extend(parse_wild_txt_file(os.path.join(step_path, "Extracted", "Wild", "Eval_Report_Wild_MultiRange.txt"), step))
    
    if ENABLE_GROUND and os.path.exists(TASK_B_OUTPUT_DIR):
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
        # 🌟 扩展为 1x5 的画板，容纳 Wild 的图表
        fig, axes = plt.subplots(1, 5, figsize=(35, 6))
        
        # [0] Summary
        df_sum = df[df["Type"] == "Summary"].copy()
        if not df_sum.empty:
            df_sum["Legend"] = df_sum["Source"] + " " + df_sum["Scene"]
            sns.lineplot(data=df_sum, x="Step", y=metric, hue="Legend", style="Legend", 
                         markers=True, linewidth=3.5, markersize=10, ax=axes[0])
            suffix = "↓" if lower_better else "↑"
            axes[0].set_title(f"Global Summary ({metric} {suffix})", fontsize=16, fontweight='bold', color='darkred')
        
        # [1] Bench
        df_bench = df[(df["Source"] == "Bench") & (df["Type"] == "Detail")]
        if not df_bench.empty:
            sns.lineplot(data=df_bench, x="Step", y=metric, hue="Scene", markers=True, ax=axes[1])
            axes[1].set_title("Bench Sub-Scenes")

        # [2] Oblique
        df_oblique = df[(df["Source"] == "Oblique") & (df["Type"] == "Detail")]
        if not df_oblique.empty:
            sns.lineplot(data=df_oblique, x="Step", y=metric, hue="Scene", markers=True, ax=axes[2])
            axes[2].set_title("Oblique Categories")
            
        # [3] Ground JSON
        df_json = df[(df["Source"] == "Depth_JSON") & (df["Type"] == "Detail")]
        if not df_json.empty:
            sns.lineplot(data=df_json, x="Step", y=metric, hue="Scene", style="Scene", markers=True, ax=axes[3])
            axes[3].set_title("Other Datasets")
            
        # [4] Wild Datasets (新增)
        df_wild = df[(df["Source"].str.startswith("Wild")) & (df["Type"] == "Detail")]
        if not df_wild.empty:
            sns.lineplot(data=df_wild, x="Step", y=metric, hue="Scene", style="Source", markers=True, ax=axes[4])
            axes[4].set_title("Wild Datasets")
            axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylabel(metric)
            ax.set_xlabel("Step")
            ax.xaxis.get_major_locator().set_params(integer=True)
        
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
    
    # 💡 核心修复：换成新的四个独立开关来判断是否需要执行 Task A 或 Task B
    if ENABLE_BENCH or ENABLE_OBLIQUE or ENABLE_WILD: 
        process_single_ckpt_task_a(SINGLE_CHECKPOINT_PATH)
        
    if ENABLE_GROUND: 
        process_single_ckpt_task_b(SINGLE_CHECKPOINT_PATH)
    
    if PLOT_ENABLE:
        print(f"📈 [Real-time Plotting] 正在生成评估对比图...")
        try: run_plotting()
        except Exception as e: print(f"⚠️ 画图失败: {e}")

    print(f"\n🎉🎉🎉 全流程执行完毕！最终评估报告与图表已生成。")