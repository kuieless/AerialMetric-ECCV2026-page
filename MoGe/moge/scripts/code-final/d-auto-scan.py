# import os
# import sys
# import subprocess
# import re
# import json
# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path

# # ================= ⚙️ 用户配置区域 =================
# PROJECT_ROOT = "/home/szq/moge2/MoGe"
# # 1. 想要扫描的权重列表 (请填入权重的绝对路径)
# CHECKPOINTS = [

#     "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00000000_ema.pt",
#     "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00001200_ema.pt",
#     "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00002400_ema.pt",

# ]

# # 2. 基础配置
# SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"  # 你的脚本所在目录
# DATA_ROOT = "/data1/szq/Val"                                  # 验证集根目录
# OUTPUT_BASE = "/data1/szq/Table2-head"                 # 所有结果的总输出目录

# # 3. 推理采样率 (0.2 代表 20%, 1.0 代表全量 100%)
# #    注意：调小这个数值可以显著加快扫描速度
# SAMPLING_RATIO = 0.2  

# # 4. 评估脚本路径 (Ground Truth 评估)
# EVAL_BASELINE_SCRIPT = "/home/szq/moge2/MoGe/moge/scripts/eval_baseline.py"
# EVAL_BASELINE_CONFIG = "configs/eval/all_benchmarks.json"

# # 5. 图表保存路径
# PLOT_SAVE_PATH = os.path.join(OUTPUT_BASE, "training_progress.png")

# # ================= 🛠️ 核心功能函数 =================

# # def run_cmd(cmd_list, log_file=None):
# #     """执行 Shell 命令"""
# #     # 将所有参数转换为字符串，防止传入 int/float 导致 subprocess 报错
# #     cmd_list = [str(x) for x in cmd_list]
# #     cmd_str = " ".join(cmd_list)
# #     print(f"⚡ Running: {cmd_str}")
    
# #     # 如果需要重定向输出到日志
# #     if log_file:
# #         with open(log_file, "w") as f:
# #             subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT)
# #     else:
# #         subprocess.run(cmd_list, check=True)
# def run_cmd(cmd_list, log_file=None, cwd=None): # <--- 1. 这里增加了 cwd=None
#     """执行 Shell 命令"""
#     # 将所有参数转换为字符串
#     cmd_list = [str(x) for x in cmd_list]
#     cmd_str = " ".join(cmd_list)
    
#     # 打印提示，如果切换了目录也显示出来
#     if cwd:
#         print(f"⚡ Running (in {cwd}): {cmd_str}")
#     else:
#         print(f"⚡ Running: {cmd_str}")
    
#     # 如果需要重定向输出到日志
#     if log_file:
#         with open(log_file, "w") as f:
#             # <--- 2. 这里传给 subprocess
#             subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=cwd) 
#     else:
#         # <--- 3. 这里传给 subprocess
#         subprocess.run(cmd_list, check=True, cwd=cwd)

# def parse_txt_report(report_path):
#     """解析 c-eval-bench.py 和 c-eval-oblique.py 生成的 txt 报告"""
#     metrics = {}
#     if not os.path.exists(report_path):
#         return None
    
#     try:
#         with open(report_path, 'r') as f:
#             lines = f.readlines()
            
#         for line in lines:
#             # 匹配 OVERALL 行
#             if "OVERALL" in line:
#                 parts = [p.strip() for p in line.split('|')]
#                 # 假设格式: Group | N | AbsRel | RMSE | ...
#                 metrics['AbsRel'] = float(parts[2])
#                 metrics['RMSE'] = float(parts[3])
#                 # metrics['a1.25'] = float(parts[5]) # 根据需要解开
#                 break
#     except Exception as e:
#         print(f"⚠️ 解析 TXT 报告出错 ({report_path}): {e}")
        
#     return metrics

# def parse_json_report(json_path):
#     """解析 eval_baseline.py 生成的 json 报告"""
#     if not os.path.exists(json_path):
#         return None
    
#     metrics = {}
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)
        
#         # 兼容列表或字典格式
#         if isinstance(data, list): 
#             data = data[0]
        
#         # 尝试寻找常见的 key
#         # 优先找 metrics 下的 abs_rel
#         if 'metrics' in data and 'abs_rel' in data['metrics']:
#              metrics['AbsRel'] = data['metrics']['abs_rel']
#         else:
#             # 备选：直接在根目录找
#             for key in ['abs_rel', 'rel', 'abs_rel_diff']:
#                 if key in data:
#                     metrics['AbsRel'] = data[key]
#                     break

#     except Exception as e:
#         print(f"⚠️ JSON 解析警告: {e}")
        
#     return metrics

# # ================= 🚀 主循环 =================

# def main():
#     results = [] 
    
#     os.makedirs(OUTPUT_BASE, exist_ok=True)
    
#     print(f"🚀 开始批量扫描，采样率设定为: {SAMPLING_RATIO * 100}%")

#     for ckpt_path in CHECKPOINTS:
#         # 1. 提取 Checkpoint 的名字和步数
#         ckpt_name = Path(ckpt_path).stem 
#         try:
#             step = int(re.search(r'\d+', ckpt_name).group())
#         except:
#             step = 0
        
#         print(f"\n{'='*60}")
#         print(f"▶️ 处理 Checkpoint: {ckpt_name} (Step: {step})")
#         print(f"{'='*60}")

#         # 定义输出目录
#         infer_out = os.path.join(OUTPUT_BASE, ckpt_name, "Infer")
#         extract_out = os.path.join(OUTPUT_BASE, ckpt_name, "Extracted")
#         eval_json_out = os.path.join(OUTPUT_BASE, ckpt_name, "ground_metrics.json")
        
#         # --- Step 1: 推理 (a-infer.py) ---
#         # 注意：如果目录已存在，默认认为已跑过。
#         # 如果你修改了采样率想重跑，请手动删除旧文件夹或修改 OUTPUT_BASE
#         if not os.path.exists(infer_out):
#             run_cmd([
#                 "python", os.path.join(SCRIPT_DIR, "a-infer.py"),
#                 "--input", DATA_ROOT,
#                 "--output", infer_out,
#                 "--model", ckpt_path,
#                 "--ratio", SAMPLING_RATIO,  # <--- 这里传入了配置的采样率
#                 "--resize", "0"
#             ])
#         else:
#             print(f"⏩ 推理结果目录已存在 ({infer_out})，跳过推理...")

#         # --- Step 2: 抽取 (b-extra.py) ---
#         infer_subdir = os.path.join(infer_out, "Val") 
#         if not os.path.exists(extract_out):
#             # 只有当 infer 成功产生了 Val 目录才抽取
#             if os.path.exists(infer_subdir):
#                 run_cmd([
#                     "python", os.path.join(SCRIPT_DIR, "b-extra.py"),
#                     "--input", infer_subdir,
#                     "--output", extract_out,
#                     "--target", "depth.npy",
#                     "--ext", ".npy"
#                 ])
#             else:
#                 print(f"⚠️ 警告: 推理子目录 {infer_subdir} 不存在，跳过抽取。")
#         else:
#             print("⏩ 抽取结果已存在，跳过...")

#         # --- Step 3: 评估 Bench & Oblique ---
#         current_result = {'step': step, 'name': ckpt_name}
        
#         # 3.1 Bench
#         bench_pred = os.path.join(extract_out, "Bench")
#         bench_gt = os.path.join(DATA_ROOT, "Bench")
        
#         if os.path.exists(bench_pred):
#             print("Running Bench Eval...")
#             run_cmd([
#                 "python", os.path.join(SCRIPT_DIR, "c-eval-bench.py"),
#                 "--pred", bench_pred,
#                 "--gt", bench_gt
#             ])
#             # 立即解析结果
#             b_metrics = parse_txt_report(os.path.join(bench_pred, "Eval_Report_Bench.txt"))
#             if b_metrics:
#                 current_result['Bench_AbsRel'] = b_metrics.get('AbsRel')
#                 current_result['Bench_RMSE'] = b_metrics.get('RMSE')
        
#         # 3.2 Oblique
#         oblique_pred = os.path.join(extract_out, "Oblique")
#         oblique_gt = os.path.join(DATA_ROOT, "Oblique")
        
#         if os.path.exists(oblique_pred):
#             print("Running Oblique Eval...")
#             run_cmd([
#                 "python", os.path.join(SCRIPT_DIR, "c-eval-oblique.py"),
#                 "--pred", oblique_pred,
#                 "--gt", oblique_gt
#             ])
#             # 立即解析结果
#             o_metrics = parse_txt_report(os.path.join(oblique_pred, "Eval_Report_Oblique_Pixel.txt"))
#             if o_metrics:
#                 current_result['Oblique_AbsRel'] = o_metrics.get('AbsRel')
#                 current_result['Oblique_RMSE'] = o_metrics.get('RMSE')

#         # --- Step 4: 评估 Ground (Baseline Script) ---
#         # 这里的采样率由 eval_baseline 内部控制（通常它不接受 ratio 参数，而是跑全量，
#         # 如果你想让 ground 也快一点，可能需要去改 eval_baseline 的源码或 config）
#         if not os.path.exists(eval_json_out):
#             print("Running Ground Baseline Eval...")
            
#             try:
#                 cmd = [
#                     "python", EVAL_BASELINE_SCRIPT,
#                     "--baseline", "baselines/moge.py",
#                     "--config", EVAL_BASELINE_CONFIG,
#                     "--output", eval_json_out,
#                     "--pretrained", ckpt_path,
#                     "--resolution_level", "9",
                    
#                     # 🔥🔥🔥 [修改] 传入比例 0.2 (即 20%)
#                     # 你也可以直接用变量: str(SAMPLING_RATIO)
#                     "--ratio", "0.2" 
#                 ]
                
#                 run_cmd(cmd, cwd=PROJECT_ROOT)

#             except subprocess.CalledProcessError as e:
#                 print(f"❌ Ground Eval 失败: {e}")
        
#         # 解析 Ground
#         g_metrics = parse_json_report(eval_json_out)
#         if g_metrics:
#             current_result['Ground_AbsRel'] = g_metrics.get('AbsRel')

#         results.append(current_result)
#         print(f"📊 当前 Step {step} 汇总: {current_result}")

#     # ================= 📈 绘图与总结 =================
    
#     if not results:
#         print("❌ 没有产生任何结果，请检查路径。")
#         return

#     df = pd.DataFrame(results)
#     df = df.sort_values(by='step')
    
#     # 保存 CSV
#     csv_path = os.path.join(OUTPUT_BASE, "all_results.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"\n💾 所有数据已保存至: {csv_path}")
    
#     # 绘图
#     plt.figure(figsize=(10, 6))
    
#     has_plot = False
#     if 'Bench_AbsRel' in df.columns and not df['Bench_AbsRel'].isnull().all():
#         plt.plot(df['step'], df['Bench_AbsRel'], marker='o', label='Bench AbsRel')
#         has_plot = True
#     if 'Oblique_AbsRel' in df.columns and not df['Oblique_AbsRel'].isnull().all():
#         plt.plot(df['step'], df['Oblique_AbsRel'], marker='s', label='Oblique AbsRel')
#         has_plot = True
#     if 'Ground_AbsRel' in df.columns and not df['Ground_AbsRel'].isnull().all():
#         plt.plot(df['step'], df['Ground_AbsRel'], marker='^', label='Ground AbsRel')
#         has_plot = True

#     if has_plot:
#         plt.title(f"Validation Metrics (Sampling Ratio: {SAMPLING_RATIO})")
#         plt.xlabel("Checkpoint Step")
#         plt.ylabel("AbsRel (Lower is Better)")
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(PLOT_SAVE_PATH)
#         print(f"📈 趋势图已保存至: {PLOT_SAVE_PATH}")
#     else:
#         print("⚠️ 数据不足，无法绘图。")

#     # 打印最佳建议
#     if 'Bench_AbsRel' in df.columns and not df['Bench_AbsRel'].isnull().all():
#         best_idx = df['Bench_AbsRel'].idxmin()
#         best_row = df.loc[best_idx]
#         print(f"\n🏆 最佳 Checkpoint (Bench): Step {best_row['step']} (AbsRel: {best_row['Bench_AbsRel']:.4f})")

# if __name__ == "__main__":
#     main()

import os
import sys
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ================= ⚙️ 用户配置区域 =================

# 1. 想要扫描的权重列表
CHECKPOINTS = [
    # "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00000000_ema.pt",
    # "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00001200_ema.pt",
    # "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00002400_ema.pt",
        # "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00003600_ema.pt",
    # "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00004800_ema.pt",
    "/home/szq/moge2/MoGe/workspace/final-neck/checkpoint/00006000_ema.pt",

]

# 2. 基础配置
SCRIPT_DIR = "/home/szq/moge2/MoGe/moge/scripts/code-final"
OUTPUT_BASE = "/data1/szq/Table2-head"

# 🔥 [修改 1] 这里不再写死 "/data1/szq/Val"，而是改成你想跑的数据集列表
TARGET_DATASETS = [
    "/data1/szq/Val/Bench",
    "/data1/szq/Val/Oblique"
]

# 3. 推理采样率
SAMPLING_RATIO = 0.2  

# 4. 图表保存路径
PLOT_SAVE_PATH = os.path.join(OUTPUT_BASE, "training_progress.png")

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
    
    print(f"🚀 开始批量扫描 (指定数据集)，采样率: {SAMPLING_RATIO * 100}%")
    print(f"📂 目标数据集: {TARGET_DATASETS}")

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
        print(f"▶️ 处理 Checkpoint: {ckpt_name} (Step: {step})")
        print(f"{'='*60}")

        # 定义输出目录
        infer_out = os.path.join(OUTPUT_BASE, ckpt_name, "Infer")
        extract_out = os.path.join(OUTPUT_BASE, ckpt_name, "Extracted")
        
        # --- Step 1: 推理 (a-infer.py) ---
        # 🔥 [修改 2] 循环每一个目标数据集，分别运行推理
        for dataset_path in TARGET_DATASETS:
            dataset_name = Path(dataset_path).name # 例如 "Bench" 或 "Oblique"
            
            # 检查这个数据集在这个ckpt下是否已经跑过
            # a-infer 的逻辑是：会在 output 目录下生成一个与输入目录同名的文件夹
            # 比如输入 .../Bench，输出就会在 .../Infer/Bench
            expected_output_dir = os.path.join(infer_out, dataset_name)
            
            if not os.path.exists(expected_output_dir):
                print(f"running inference for {dataset_name}...")
                run_cmd([
                    "python", os.path.join(SCRIPT_DIR, "a-infer.py"),
                    "--input", dataset_path,  # 只传入 Bench 或 Oblique
                    "--output", infer_out,    # 输出到 Infer 根目录
                    "--model", str(ckpt_path),
                    "--ratio", SAMPLING_RATIO,
                    "--resize", "0"
                ])
            else:
                print(f"⏩ {dataset_name} 推理结果已存在，跳过...")

        # --- Step 2: 抽取 (b-extra.py) ---
        # 🔥 [修改 3] 直接把 infer_out 传进去，不加 "Val"
        # 因为我们分开跑后，目录结构变成了 Infer/Bench 和 Infer/Oblique
        # b-extra.py 会自动递归扫描里面所有的 depth.npy，所以直接传 Infer 根目录最稳妥
        if not os.path.exists(extract_out):
            if os.path.exists(infer_out):
                run_cmd([
                    "python", os.path.join(SCRIPT_DIR, "b-extra.py"),
                    "--input", infer_out,    # <--- 改成了 infer_out
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
        bench_gt = os.path.join("/data1/szq/Val", "Bench") # 这里最好写死绝对路径，或者从 TARGET_DATASETS 推断
        
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
    
    csv_path = os.path.join(OUTPUT_BASE, "bench_oblique_results.csv")
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
        plt.title(f"Validation Metrics (Sampling Ratio: {SAMPLING_RATIO})")
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