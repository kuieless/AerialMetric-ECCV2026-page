import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json

# ================= 1. 配置区域 =================

# 🔥 请修改你的根目录
ROOT_DIR = "/data1/szq/Table2-head-LoRA-16-Ground"

# ================= 2. 解析逻辑 (JSON版) =================

def parse_json_file(file_path, step):
    data_rows = []
    
    if not os.path.exists(file_path):
        return data_rows

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        # 遍历 JSON 中的每一个数据集 (NYUv2, KITTI, ..., mean)
        for dataset_name, metrics_group in content.items():
            
            # 我们只关心 depth_metric
            if "depth_metric" not in metrics_group:
                continue
            
            metrics = metrics_group["depth_metric"]
            
            # 提取指标，带默认值防止报错
            # rel -> AbsRel
            # rmse -> RMSE
            # delta1 -> a1.25
            abs_rel = metrics.get("rel", None)
            rmse = metrics.get("rmse", None)
            a125 = metrics.get("delta1", None)
            
            if abs_rel is None: continue # 如果关键数据缺失则跳过

            data_rows.append({
                "Step": int(step),
                "Dataset": dataset_name,
                "AbsRel": float(abs_rel),
                "RMSE": float(rmse),
                "a1.25": float(a125),
                "Is_Mean": (dataset_name.lower() == "mean")
            })
            
    except Exception as e:
        print(f"⚠️ 读取出错 {file_path}: {e}")
        return []
            
    return data_rows

def scan_directory(root_dir):
    all_data = []
    if not os.path.exists(root_dir):
        print(f"❌ 错误: 路径不存在 {root_dir}")
        return []

    subdirs = os.listdir(root_dir)
    found_steps = []
    
    print(f"📂 正在扫描: {root_dir} ...")
    
    for subdir in subdirs:
        # 检查是否是纯数字文件夹 (例如 00000100)
        if subdir.isdigit(): 
            step = int(subdir)
            step_path = os.path.join(root_dir, subdir)
            json_file = os.path.join(step_path, "metrics.json")
            
            # 解析该 Step 的 metrics.json
            rows = parse_json_file(json_file, step)
            if rows:
                all_data.extend(rows)
                found_steps.append(step)
    
    if found_steps:
        print(f"✅ 扫描完成，共找到 Step: {sorted(found_steps)}")
    else:
        print("❌ 未找到任何有效的数据文件 (metrics.json)")
        
    return all_data

# ================= 3. 绘图逻辑 =================

def main():
    data = scan_directory(ROOT_DIR)
    if not data: return

    df = pd.DataFrame(data)
    df = df.sort_values(by="Step")

    # 设置 Seaborn 主题
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 核心绘图函数
    def plot_metric_json(metric_name, filename, lower_is_better=True):
        print(f"🎨 绘制: {metric_name} ...")
        
        # 创建 1行2列 的画布 
        # 左边: Mean (Total)
        # 右边: 各个数据集 (Datasets)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # --- 子图 1: Mean ---
        df_mean = df[df["Is_Mean"] == True]
        if not df_mean.empty:
            sns.lineplot(
                data=df_mean, 
                x="Step", y=metric_name, hue="Dataset", 
                marker="o", markersize=10, linewidth=3.5, # 加粗
                palette=["#d62728"], # 红色突出
                ax=axes[0]
            )
            suffix = "↓" if lower_is_better else "↑"
            axes[0].set_title(f"Mean Performance ({metric_name} {suffix})", fontsize=16, fontweight='bold')
            axes[0].set_ylabel(metric_name)
            axes[0].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[0].set_title("Mean Data Not Found")

        # --- 子图 2: Individual Datasets (排除 Mean) ---
        df_others = df[df["Is_Mean"] == False]
        if not df_others.empty:
            sns.lineplot(
                data=df_others, 
                x="Step", y=metric_name, hue="Dataset", style="Dataset",
                markers=True, dashes=False, 
                markersize=8, linewidth=2.0,
                ax=axes[1]
            )
            axes[1].set_title(f"Individual Datasets ({metric_name})", fontsize=16)
            axes[1].set_ylabel(metric_name)
            axes[1].legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放外边防止挡住
            axes[1].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[1].set_title("No Individual Datasets Found")

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"   -> 保存至: {filename}")

    # 分别绘制三个指标
    # 1. AbsRel (越低越好)
    plot_metric_json("AbsRel", "ground_plot_absrel.png", lower_is_better=True)
    
    # 2. RMSE (越低越好)
    plot_metric_json("RMSE", "ground_plot_rmse.png", lower_is_better=True)
    
    # 3. a1.25 (越高越好)
    plot_metric_json("a1.25", "ground_plot_a125.png", lower_is_better=False)

if __name__ == "__main__":
    main()