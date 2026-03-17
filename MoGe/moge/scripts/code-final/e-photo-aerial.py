import matplotlib.pyplot as plt
import pandas as pd
import io
import seaborn as sns

# ================= 1. 模拟数据 =================

BENCH_TXT_00000 = """
Group/Scene                  | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
-----------------------------------------------------------------------------------------------------------------------------
OVERALL                      | 927  | 0.3371  | 41.037  | 0.140  | 0.330  | 0.039  | 0.918  | 0.074 
Cleaned_Dataset_Campus       | 176  | 0.2376  | 26.796  | 0.154  | 0.424  | 0.060  | 0.959  | 0.063 
Cleaned_Dataset_Factory      | 190  | 0.2252  | 26.262  | 0.211  | 0.445  | 0.045  | 0.961  | 0.057 
Cleaned_Dataset_Farm         | 190  | 0.6795  | 87.784  | 0.000  | 0.000  | 0.025  | 0.807  | 0.088 
Cleaned_Dataset_Gress        | 371  | 0.2662  | 31.418  | 0.170  | 0.395  | 0.033  | 0.934  | 0.082 
"""

OBLIQUE_TXT_00000 = """
Category/Scene         | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
------------------------------------------------------------------------------------------------------------------------
OVERALL                | 296  | 0.4939  | 70.601  | 0.037  | 0.088  | 0.049  | 0.881  | 0.114 
[CAT] City             | 172  | 0.4900  | 87.070  | 0.011  | 0.042  | 0.052  | 0.947  | 0.086 
[CAT] Natural          | 65   | 0.4475  | 43.654  | 0.139  | 0.290  | 0.051  | 0.747  | 0.155 
[CAT] Rural            | 59   | 0.5565  | 52.277  | 0.000  | 0.000  | 0.041  | 0.840  | 0.149 
"""

BENCH_TXT_01200 = """
Group/Scene                  | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
-----------------------------------------------------------------------------------------------------------------------------
OVERALL                      | 927  | 0.2631  | 31.933  | 0.132  | 0.368  | 0.038  | 0.918  | 0.074 
-----------------------------------------------------------------------------------------------------------------------------
>>> By Scene Dataset:
Cleaned_Dataset_Campus       | 176  | 0.2307  | 25.971  | 0.141  | 0.377  | 0.060  | 0.959  | 0.063 
Cleaned_Dataset_Factory      | 190  | 0.2041  | 23.097  | 0.192  | 0.499  | 0.045  | 0.961  | 0.057 
Cleaned_Dataset_Farm         | 190  | 0.4345  | 56.410  | 0.005  | 0.072  | 0.024  | 0.809  | 0.086 
Cleaned_Dataset_Gress        | 371  | 0.2211  | 26.750  | 0.162  | 0.447  | 0.031  | 0.934  | 0.082 
"""

OBLIQUE_TXT_01200 = """
Category/Scene         | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
------------------------------------------------------------------------------------------------------------------------
OVERALL                | 296  | 0.3462  | 49.841  | 0.069  | 0.186  | 0.049  | 0.881  | 0.114 
------------------------------------------------------------------------------------------------------------------------
[CAT] City             | 172  | 0.3513  | 62.080  | 0.041  | 0.173  | 0.052  | 0.947  | 0.086 
[CAT] Natural          | 65   | 0.2854  | 28.047  | 0.207  | 0.367  | 0.051  | 0.746  | 0.154 
[CAT] Rural            | 59   | 0.3984  | 38.174  | 0.000  | 0.021  | 0.041  | 0.840  | 0.149 
"""
BENCH_TXT_02400 = """
Group/Scene                  | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
-----------------------------------------------------------------------------------------------------------------------------
OVERALL                      | 927  | 0.2114  | 25.343  | 0.193  | 0.488  | 0.038  | 0.918  | 0.074 
-----------------------------------------------------------------------------------------------------------------------------
>>> By Scene Dataset:
Cleaned_Dataset_Campus       | 176  | 0.2132  | 23.501  | 0.173  | 0.447  | 0.060  | 0.959  | 0.063 
Cleaned_Dataset_Factory      | 190  | 0.1881  | 20.874  | 0.240  | 0.554  | 0.045  | 0.961  | 0.057 
Cleaned_Dataset_Farm         | 190  | 0.2661  | 34.832  | 0.172  | 0.373  | 0.024  | 0.808  | 0.085 
Cleaned_Dataset_Gress        | 371  | 0.1945  | 23.644  | 0.190  | 0.533  | 0.031  | 0.934  | 0.082 
"""

OBLIQUE_TXT_02400 = """
Category/Scene         | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
------------------------------------------------------------------------------------------------------------------------
OVERALL                | 296  | 0.2690  | 38.696  | 0.127  | 0.364  | 0.049  | 0.882  | 0.114 
------------------------------------------------------------------------------------------------------------------------
[CAT] City             | 172  | 0.2775  | 48.791  | 0.122  | 0.361  | 0.052  | 0.947  | 0.086 
[CAT] Natural          | 65   | 0.1975  | 18.549  | 0.219  | 0.515  | 0.051  | 0.746  | 0.154 
[CAT] Rural            | 59   | 0.3231  | 31.464  | 0.042  | 0.208  | 0.041  | 0.841  | 0.149 
"""






BENCH_TXT_04800 = """
Group/Scene                  | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
-----------------------------------------------------------------------------------------------------------------------------
OVERALL                      | 927  | 0.1756  | 20.858  | 0.266  | 0.614  | 0.038  | 0.918  | 0.074 
-----------------------------------------------------------------------------------------------------------------------------
>>> By Scene Dataset:
Cleaned_Dataset_Campus       | 176  | 0.1810  | 19.372  | 0.235  | 0.581  | 0.060  | 0.959  | 0.063 
Cleaned_Dataset_Factory      | 190  | 0.1639  | 18.065  | 0.310  | 0.657  | 0.045  | 0.961  | 0.057 
Cleaned_Dataset_Farm         | 190  | 0.1961  | 25.537  | 0.272  | 0.574  | 0.024  | 0.808  | 0.085 
Cleaned_Dataset_Gress        | 371  | 0.1685  | 20.597  | 0.256  | 0.629  | 0.031  | 0.934  | 0.082 
"""

OBLIQUE_TXT_04800 = """
Category/Scene         | N    | AbsRel  | RMSE    | a1.10  | a1.25  | SI-Log | Spear  | N-RMSE
------------------------------------------------------------------------------------------------------------------------
OVERALL                | 296  | 0.2310  | 33.393  | 0.171  | 0.526  | 0.049  | 0.881  | 0.114 
------------------------------------------------------------------------------------------------------------------------
[CAT] City             | 172  | 0.2438  | 42.837  | 0.159  | 0.492  | 0.052  | 0.946  | 0.086 
[CAT] Natural          | 65   | 0.1587  | 14.172  | 0.260  | 0.735  | 0.051  | 0.747  | 0.154 
[CAT] Rural            | 59   | 0.2734  | 27.034  | 0.107  | 0.396  | 0.041  | 0.841  | 0.149 
"""
# ================= 2. 解析逻辑 =================

def parse_report_content(text_content, dataset_name, step):
    data_rows = []
    lines = text_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if "|" not in line or "---" in line or "Group/Scene" in line or "Category/Scene" in line:
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue # 确保有足够的列
        
        scene_name = parts[0]
        try:
            # 🔥 [新增] 解析多个指标
            # AbsRel: index 2
            # RMSE: index 3
            # a1.25: index 5
            abs_rel = float(parts[2])
            rmse = float(parts[3])
            a125 = float(parts[5])
            
            # 美化一下名字
            clean_name = scene_name.replace("Cleaned_Dataset_", "").replace("[CAT] ", "")
            
            data_rows.append({
                "Step": step,
                "Dataset": dataset_name,
                "Scene": clean_name,
                "AbsRel": abs_rel,
                "RMSE": rmse,
                "a1.25": a125,
                "Is_Overall": (clean_name == "OVERALL")
            })
        except ValueError:
            continue
            
    return data_rows

# ================= 3. 主程序 =================

def main():
    all_data = []
    # 使用修正后的变量名
    all_data.extend(parse_report_content(BENCH_TXT_00000, "Bench", 0))
    all_data.extend(parse_report_content(OBLIQUE_TXT_00000, "Oblique", 0))
    all_data.extend(parse_report_content(BENCH_TXT_01200, "Bench", 1200))
    all_data.extend(parse_report_content(OBLIQUE_TXT_01200, "Oblique", 1200))
    
    df = pd.DataFrame(all_data)
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # 定义核心绘图函数
    # ---------------------------------------------------------
    def plot_metric(metric_name, filename, lower_is_better=True):
        print(f"绘制: {filename} ...")
        
        # 创建一个包含 3 个子图的大图 (1行3列)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        suffix = "(Lower is Better)" if lower_is_better else "(Higher is Better)"
        
        # 子图 1: Overall 对比
        df_overall = df[df["Is_Overall"] == True]
        sns.lineplot(data=df_overall, x="Step", y=metric_name, hue="Dataset", marker="o", linewidth=2.5, ax=axes[0])
        axes[0].set_title(f"Overall {metric_name} {suffix}", fontsize=12, fontweight='bold')
        axes[0].set_ylabel(metric_name)
        
        # 子图 2: Bench 子场景
        df_scenes = df[df["Is_Overall"] == False]
        sns.lineplot(
            data=df_scenes[df_scenes["Dataset"] == "Bench"], 
            x="Step", y=metric_name, hue="Scene", marker="s", ax=axes[1]
        )
        axes[1].set_title(f"Bench Sub-Scenes {metric_name}", fontsize=12)
        axes[1].set_ylabel(metric_name)
        
        # 子图 3: Oblique 子场景
        sns.lineplot(
            data=df_scenes[df_scenes["Dataset"] == "Oblique"], 
            x="Step", y=metric_name, hue="Scene", marker="^", ax=axes[2]
        )
        axes[2].set_title(f"Oblique Categories {metric_name}", fontsize=12)
        axes[2].set_ylabel(metric_name)
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"✅ 生成完毕: {filename}")

    # ---------------------------------------------------------
    # 分别调用三次
    # ---------------------------------------------------------
    
    # 1. AbsRel (越低越好)
    plot_metric("AbsRel", "demo_plot_metric_absrel.png", lower_is_better=True)
    
    # 2. RMSE (越低越好)
    plot_metric("RMSE", "demo_plot_metric_rmse.png", lower_is_better=True)
    
    # 3. a1.25 (越高越好)
    plot_metric("a1.25", "demo_plot_metric_a125.png", lower_is_better=False)

if __name__ == "__main__":
    main()