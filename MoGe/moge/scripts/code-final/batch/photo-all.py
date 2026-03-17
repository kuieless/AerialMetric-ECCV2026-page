import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json

# ================= 1. 配置区域 =================

# 🔥 路径配置
# 1. 存放 Bench/Oblique 报告的根目录 (包含 00000100/Extracted/...)
ROOT_DIR_TXT = "/data1/szq/Infer-Final/Table2/Table2-head-LoRA-32"

# 2. 存放 Depth Metrics JSON 的根目录 (包含 00000100/metrics.json)
ROOT_DIR_JSON = "/data1/szq/Table2-head-LoRA-32-Ground"

# ================= 2. 数据解析逻辑 =================

# --- A. 解析 TXT 报告 (Bench & Oblique) ---

BENCH_REL_PATH = os.path.join("Extracted", "Bench", "Eval_Report_Bench.txt")
OBLIQUE_REL_PATH = os.path.join("Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt")

def clean_txt_scene_name(raw_name):
    name = raw_name.strip()
    if "Cleaned_Dataset_" in name: return name.replace("Cleaned_Dataset_", "")
    if "[CAT]" in name: return name.replace("[CAT]", "").strip()
    return name

def parse_txt_file(file_path, source_type, step):
    rows = []
    if not os.path.exists(file_path): return rows

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 白名单过滤
    if source_type == "Bench":
        targets = ["OVERALL", "Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory", "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress"]
    else: # Oblique
        targets = ["OVERALL", "[CAT] City", "[CAT] Natural", "[CAT] Rural"]

    for line in lines:
        if "|" not in line or "---" in line: continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        
        raw_name = parts[0]
        # 匹配白名单
        if raw_name not in targets: continue

        try:
            abs_rel = float(parts[2])
            rmse = float(parts[3])
            a125 = float(parts[5])
            scene = clean_txt_scene_name(raw_name)
            
            # 标记是否是总分
            is_summary = (scene.upper() == "OVERALL")
            
            rows.append({
                "Step": step,
                "Source": source_type, # Bench or Oblique
                "Scene": scene,
                "AbsRel": abs_rel,
                "RMSE": rmse,
                "a1.25": a125,
                "Type": "Summary" if is_summary else "Detail"
            })
        except ValueError: continue
    return rows

# --- B. 解析 JSON 报告 (Depth Metrics) ---

def parse_json_file(file_path, step):
    rows = []
    if not os.path.exists(file_path): return rows

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        for dataset_name, metrics_group in content.items():
            if "depth_metric" not in metrics_group: continue
            
            metrics = metrics_group["depth_metric"]
            abs_rel = metrics.get("rel", None)
            rmse = metrics.get("rmse", None)
            a125 = metrics.get("delta1", None)
            
            if abs_rel is None: continue

            # 标记是否是总分 (mean)
            is_summary = (dataset_name.lower() == "mean")
            
            rows.append({
                "Step": step,
                "Source": "Depth_JSON",
                "Scene": "Mean" if is_summary else dataset_name,
                "AbsRel": float(abs_rel),
                "RMSE": float(rmse),
                "a1.25": float(a125),
                "Type": "Summary" if is_summary else "Detail"
            })
    except Exception as e:
        print(f"JSON Error in {step}: {e}")
    return rows

# --- C. 统一扫描入口 ---

def collect_all_data():
    all_data = []
    
    # 1. 扫描 TXT 目录
    print(f"📂 扫描 TXT 根目录: {ROOT_DIR_TXT}")
    if os.path.exists(ROOT_DIR_TXT):
        for subdir in os.listdir(ROOT_DIR_TXT):
            if subdir.isdigit():
                step = int(subdir)
                step_path = os.path.join(ROOT_DIR_TXT, subdir)
                # Parse Bench
                all_data.extend(parse_txt_file(os.path.join(step_path, BENCH_REL_PATH), "Bench", step))
                # Parse Oblique
                all_data.extend(parse_txt_file(os.path.join(step_path, OBLIQUE_REL_PATH), "Oblique", step))
    else:
        print(f"❌ TXT 目录不存在: {ROOT_DIR_TXT}")

    # 2. 扫描 JSON 目录
    print(f"📂 扫描 JSON 根目录: {ROOT_DIR_JSON}")
    if os.path.exists(ROOT_DIR_JSON):
        for subdir in os.listdir(ROOT_DIR_JSON):
            if subdir.isdigit():
                step = int(subdir)
                json_path = os.path.join(ROOT_DIR_JSON, subdir, "metrics.json")
                all_data.extend(parse_json_file(json_path, step))
    else:
        print(f"❌ JSON 目录不存在: {ROOT_DIR_JSON}")
        
    return all_data

# ================= 3. 绘图逻辑 =================

def main():
    raw_data = collect_all_data()
    if not raw_data:
        print("未获取到任何数据，终止。")
        return

    df = pd.DataFrame(raw_data)
    df = df.sort_values(by="Step")
    
    # 打印一下找到的数据概览
    print(f"✅ 数据加载完成，共 {len(df)} 条记录。")
    print(f"   涉及 Steps: {df['Step'].unique().tolist()}")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 核心绘图函数
    def plot_combined_metric(metric_name, filename, lower_is_better=True):
        print(f"🎨 正在绘制: {metric_name} -> {filename}")
        
        # 定义 1 行 4 列的画布
        fig, axes = plt.subplots(1, 4, figsize=(28, 6))
        
        # -------------------------------------------------
        # [子图 1] Summary (Three Overalls) - 您的核心需求
        # -------------------------------------------------
        # 筛选: Bench的Overall, Oblique的Overall, JSON的Mean
        df_summary = df[df["Type"] == "Summary"].copy()
        
        # 为了图例好看，组合 Source 和 Scene
        # 结果如: "Bench Overall", "Oblique Overall", "Depth_JSON Mean"
        df_summary["Legend_Label"] = df_summary["Source"] + " " + df_summary["Scene"]
        
        if not df_summary.empty:
            sns.lineplot(
                data=df_summary, x="Step", y=metric_name, hue="Legend_Label",
                style="Legend_Label", markers=True, dashes=False,
                linewidth=3.5, markersize=10, ax=axes[0]
            )
            suffix = "↓" if lower_is_better else "↑"
            axes[0].set_title(f"⭐ Global Summary ({metric_name} {suffix})", fontsize=16, fontweight='bold', color='darkred')
            axes[0].set_ylabel(metric_name)
            axes[0].legend(title="Overview", loc='best')
        else:
            axes[0].set_title("No Summary Data")

        # -------------------------------------------------
        # [子图 2] Bench Details
        # -------------------------------------------------
        df_bench = df[(df["Source"] == "Bench") & (df["Type"] == "Detail")]
        if not df_bench.empty:
            sns.lineplot(
                data=df_bench, x="Step", y=metric_name, hue="Scene",
                markers=True, dashes=False, linewidth=2, ax=axes[1]
            )
            axes[1].set_title(f"Bench Sub-Scenes", fontsize=14)
        else:
            axes[1].set_title("No Bench Details")

        # -------------------------------------------------
        # [子图 3] Oblique Details
        # -------------------------------------------------
        df_oblique = df[(df["Source"] == "Oblique") & (df["Type"] == "Detail")]
        if not df_oblique.empty:
            sns.lineplot(
                data=df_oblique, x="Step", y=metric_name, hue="Scene",
                markers=True, dashes=False, linewidth=2, ax=axes[2]
            )
            axes[2].set_title(f"Oblique Categories", fontsize=14)
        else:
            axes[2].set_title("No Oblique Details")

        # -------------------------------------------------
        # [子图 4] Depth JSON Details
        # -------------------------------------------------
        df_json = df[(df["Source"] == "Depth_JSON") & (df["Type"] == "Detail")]
        if not df_json.empty:
            sns.lineplot(
                data=df_json, x="Step", y=metric_name, hue="Scene",
                style="Scene", markers=True, dashes=False, linewidth=2, ax=axes[3]
            )
            axes[3].set_title(f"Other Datasets (JSON)", fontsize=14)
            # JSON 数据集通常较多，图例放外侧防止遮挡
            axes[3].legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[3].set_title("No JSON Details")

        # 全局美化
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel("Training Step")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight') # bbox_inches防止图例被切
        print(f"   -> 保存成功")

    # 执行绘图
    plot_combined_metric("AbsRel", "compare_absrel.png", lower_is_better=True)
    plot_combined_metric("RMSE", "compare_rmse.png", lower_is_better=True)
    plot_combined_metric("a1.25", "compare_a125.png", lower_is_better=False)

if __name__ == "__main__":
    main()