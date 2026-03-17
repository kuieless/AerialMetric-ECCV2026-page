import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ================= 1. 配置区域 =================

# 🔥 请修改根目录
ROOT_DIR = "/data1/szq/Infer-Final/Table2/Table2-head-LoRA-16-32"

# 定义相对路径
BENCH_REL_PATH = os.path.join("Extracted", "Bench", "Eval_Report_Bench.txt")
OBLIQUE_REL_PATH = os.path.join("Extracted", "Oblique", "Eval_Report_Oblique_Pixel.txt")

# ================= 2. 解析逻辑 (精确过滤) =================

def clean_scene_name(raw_name):
    """
    美化名字，去掉冗余前缀
    """
    name = raw_name.strip()
    if "Cleaned_Dataset_" in name:
        return name.replace("Cleaned_Dataset_", "")
    if "[CAT]" in name:
        return name.replace("[CAT]", "").strip()
    if name == "OVERALL":
        return "Overall"
    return name

def parse_report_file(file_path, dataset_type, step):
    data_rows = []
    if not os.path.exists(file_path):
        return data_rows

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 定义需要的场景关键词 (白名单)
    # 只要行里包含这些词，就保留
    target_scenes = []
    if dataset_type == "Bench":
        target_scenes = ["OVERALL", "Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory", "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress"]
    else: # Oblique
        target_scenes = ["OVERALL", "[CAT] City", "[CAT] Natural", "[CAT] Rural"]

    for line in lines:
        line = line.strip()
        if "|" not in line or "---" in line: continue

        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        
        raw_name = parts[0]
        
        # 检查是否是我们需要的行
        is_target = False
        for t in target_scenes:
            if t == raw_name: # 精确匹配名字
                is_target = True
                break
        
        if not is_target:
            continue

        try:
            # 解析数值
            # AbsRel: index 2, RMSE: index 3, a1.25: index 5
            abs_rel = float(parts[2])
            rmse = float(parts[3])
            a125 = float(parts[5])
            
            clean_name = clean_scene_name(raw_name)
            
            data_rows.append({
                "Step": int(step),
                "Dataset": dataset_type,
                "Scene": clean_name,
                "AbsRel": abs_rel,
                "RMSE": rmse,
                "a1.25": a125
            })
        except ValueError:
            continue
            
    return data_rows

def scan_directory(root_dir):
    all_data = []
    if not os.path.exists(root_dir):
        print(f"❌ 错误: 路径不存在 {root_dir}")
        return []

    subdirs = os.listdir(root_dir)
    found_steps = []
    
    for subdir in subdirs:
        if subdir.isdigit(): # 只要纯数字文件夹
            step = int(subdir)
            found_steps.append(step)
            step_path = os.path.join(root_dir, subdir)
            
            # 读取 Bench
            bench_file = os.path.join(step_path, BENCH_REL_PATH)
            all_data.extend(parse_report_file(bench_file, "Bench", step))
            
            # 读取 Oblique
            oblique_file = os.path.join(step_path, OBLIQUE_REL_PATH)
            all_data.extend(parse_report_file(oblique_file, "Oblique", step))
    
    print(f"✅ 扫描完成，共找到 Step: {sorted(found_steps)}")
    return all_data

# ================= 3. 绘图逻辑 (美化版) =================

def main():
    data = scan_directory(ROOT_DIR)
    if not data: return

    df = pd.DataFrame(data)
    df = df.sort_values(by="Step")

    # 设置 Seaborn 主题
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 定义绘图核心函数
    def plot_beautiful_metric(metric_name, filename, lower_is_better=True):
        print(f"🎨 绘制: {metric_name} ...")
        
        # 创建 1行2列 的画布 (左边 Bench, 右边 Oblique)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        datasets = ["Bench", "Oblique"]
        
        # 遍历两个数据集分别画图
        for i, ds_name in enumerate(datasets):
            ax = axes[i]
            subset = df[df["Dataset"] == ds_name]
            
            if subset.empty:
                ax.set_title(f"{ds_name} - No Data")
                continue

            # 画折线图
            # style="Scene" 会自动分配不同的虚线/实线
            # markers=True 会自动分配不同的点形状
            sns.lineplot(
                data=subset, 
                x="Step", 
                y=metric_name, 
                hue="Scene", 
                style="Scene", 
                markers=True, 
                dashes=False, # 全部用实线，只靠颜色和Marker区分，看起来更整洁
                markersize=9,
                linewidth=2.5,
                ax=ax
            )
            
            # 标题与标签
            suffix = "↓" if lower_is_better else "↑"
            ax.set_title(f"{ds_name} Dataset ({metric_name} {suffix})", fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel("Training Steps", fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            
            # 优化图例 (Legend)
            ax.legend(title="Scenes", title_fontsize='13', fontsize='11', loc='best')
            
            # 优化 Grid
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(filename, dpi=300) # 300 dpi 更清晰
        print(f"   -> 保存至: {filename}")

    # 执行绘制
    plot_beautiful_metric("AbsRel", "plot_pretty_absrel.png", lower_is_better=True)
    plot_beautiful_metric("RMSE", "plot_pretty_rmse.png", lower_is_better=True)
    plot_beautiful_metric("a1.25", "plot_pretty_a125.png", lower_is_better=False)

if __name__ == "__main__":
    main()