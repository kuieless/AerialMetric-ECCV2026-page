
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from math import pi
import warnings
warnings.filterwarnings("ignore")

# ================= 学术绘图全局设置 =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.linewidth': 1.0
})

COLOR_BASE = '#E64B35' # Science 红
COLOR_OURS = '#4DBBD5' # Science 蓝

# ================= 1. 数据读取与处理 =================
def load_data(baseline_csv, ours_csv):
    df_base = pd.read_csv(baseline_csv)
    df_ours = pd.read_csv(ours_csv)
    
    scene_map = {
        'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
        'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
    }
    df_base['Scene'] = df_base['Scene'].map(scene_map)
    df_ours['Scene'] = df_ours['Scene'].map(scene_map)
    
    metrics = ['a1.25'] # 这次只关注核心指标
    keys = ['Filename', 'Scene', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    
    df_b = df_base[keys + metrics].rename(columns={'a1.25': 'a1.25_Base'})
    df_o = df_ours[keys + metrics].rename(columns={'a1.25': 'a1.25_Ours'})
    
    df = pd.merge(df_b, df_o, on=keys, how='inner')
    if df['a1.25_Base'].max() <= 1.05:
        df['a1.25_Base'] *= 100
        df['a1.25_Ours'] *= 100
        
    return df

# ================= 2. 核心多子图绘制引擎 =================
def plot_scene_wise_matrix(df, save_path):
    scenes = ['Campus', 'Factory', 'Farm', 'Grass']
    
    # 创建 4行(场景) x 3列(图表类型) 的大网格
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(4, 3, wspace=0.25, hspace=0.35)
    
    # 预处理长格式数据供 Seaborn 使用
    df_melt = pd.melt(df, id_vars=['Scene', 'Pitch_Group', 'Height_Group', 'FOV_Group'], 
                      value_vars=['a1.25_Base', 'a1.25_Ours'], 
                      var_name='Model', value_name='a1.25_Score')
    df_melt['Model'] = df_melt['Model'].map({'a1.25_Base': 'Baseline', 'a1.25_Ours': 'Ours'})
    
    for i, scene in enumerate(scenes):
        df_scene = df[df['Scene'] == scene]
        df_melt_scene = df_melt[df_melt['Scene'] == scene]
        
        # ---------------------------------------------------------
        # 列 1: 放弃小提琴图，换成极度清晰的箱线图 (Pitch vs Accuracy)
        # ---------------------------------------------------------
        ax1 = fig.add_subplot(gs[i, 0])
        sns.boxplot(data=df_melt_scene, x='Pitch_Group', y='a1.25_Score', hue='Model', 
                    palette={'Baseline': COLOR_BASE, 'Ours': COLOR_OURS},
                    showfliers=False, width=0.6, ax=ax1) # showfliers=False 隐藏离群点让主体更清晰
        
        ax1.set_title(f"[{scene}] Pitch-wise Distribution", weight='bold')
        ax1.set_ylabel(r"Accuracy $a_{1.25}$ (%)")
        ax1.set_xlabel("Camera Pitch Angle (°)")
        ax1.set_ylim(-5, 105)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        if i != 0: ax1.legend_.remove() # 只在第一行保留图例
        else: ax1.legend(loc='lower right')

        # ---------------------------------------------------------
        # 列 2: 全因子鲁棒性雷达图 (8 个属性维度)
        # ---------------------------------------------------------
        ax2 = fig.add_subplot(gs[i, 1], polar=True)
        
        # 定义 8 个因子方向
        factors = ['P=-90°', 'P=-75°', 'P=-60°', 'P=-45°', 'H=80m', 'H=120m', 'FOV=63°', 'FOV=83°']
        
        # 计算 Baseline 和 Ours 在这 8 个因子上的各自平均精度
        p90_b = df_scene[df_scene['Pitch_Group'] == -90]['a1.25_Base'].mean()
        p75_b = df_scene[df_scene['Pitch_Group'] == -75]['a1.25_Base'].mean()
        p60_b = df_scene[df_scene['Pitch_Group'] == -60]['a1.25_Base'].mean()
        p45_b = df_scene[df_scene['Pitch_Group'] == -45]['a1.25_Base'].mean()
        h80_b = df_scene[df_scene['Height_Group'] == 80]['a1.25_Base'].mean()
        h120_b = df_scene[df_scene['Height_Group'] == 120]['a1.25_Base'].mean()
        fov63_b = df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].min()]['a1.25_Base'].mean()
        fov83_b = df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].max()]['a1.25_Base'].mean()
        vals_base = [p90_b, p75_b, p60_b, p45_b, h80_b, h120_b, fov63_b, fov83_b]
        
        p90_o = df_scene[df_scene['Pitch_Group'] == -90]['a1.25_Ours'].mean()
        p75_o = df_scene[df_scene['Pitch_Group'] == -75]['a1.25_Ours'].mean()
        p60_o = df_scene[df_scene['Pitch_Group'] == -60]['a1.25_Ours'].mean()
        p45_o = df_scene[df_scene['Pitch_Group'] == -45]['a1.25_Ours'].mean()
        h80_o = df_scene[df_scene['Height_Group'] == 80]['a1.25_Ours'].mean()
        h120_o = df_scene[df_scene['Height_Group'] == 120]['a1.25_Ours'].mean()
        fov63_o = df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].min()]['a1.25_Ours'].mean()
        fov83_o = df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].max()]['a1.25_Ours'].mean()
        vals_ours = [p90_o, p75_o, p60_o, p45_o, h80_o, h120_o, fov63_o, fov83_o]
        
        # 雷达图闭合
        angles = [n / float(len(factors)) * 2 * pi for n in range(len(factors))]
        vals_base += vals_base[:1]
        vals_ours += vals_ours[:1]
        angles += angles[:1]
        
        ax2.set_theta_offset(pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(factors, weight='bold', size=9)
        ax2.set_ylim(0, 100) # 将最高分固定在 100
        ax2.set_yticks([20, 40, 60, 80])
        ax2.set_yticklabels(['20', '40', '60', '80'], color="grey", size=7)
        
        ax2.plot(angles, vals_base, color=COLOR_BASE, linewidth=1.5, linestyle='solid')
        ax2.fill(angles, vals_base, color=COLOR_BASE, alpha=0.15)
        ax2.plot(angles, vals_ours, color=COLOR_OURS, linewidth=2, linestyle='solid')
        ax2.fill(angles, vals_ours, color=COLOR_OURS, alpha=0.3)
        ax2.set_title(f"[{scene}] Factor Robustness Radar", weight='bold', pad=15)

        # ---------------------------------------------------------
        # 列 3: Pitch 与 Height 交叉效应折线图 (保留解耦分析)
        # ---------------------------------------------------------
        ax3 = fig.add_subplot(gs[i, 2])
        sns.lineplot(data=df_scene, x='Pitch_Group', y='a1.25_Base', hue='Height_Group', 
                     palette={80: '#ff9999', 120: '#8b0000'}, linestyle='--', marker='o', ax=ax3, legend=False)
        sns.lineplot(data=df_scene, x='Pitch_Group', y='a1.25_Ours', hue='Height_Group', 
                     palette={80: '#99ccff', 120: '#00008b'}, linestyle='-', marker='s', ax=ax3, legend=False)
                     
        ax3.set_title(f"[{scene}] Pitch & Height Disentanglement", weight='bold')
        ax3.set_xlabel("Camera Pitch Angle (°)")
        ax3.set_ylabel(r"Accuracy $a_{1.25}$ (%)")
        ax3.set_xticks([-90, -75, -60, -45])
        ax3.set_ylim(-5, 105)
        ax3.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='#8b0000', linestyle='--', marker='o', label='Base (120m)'),
                Line2D([0], [0], color='#ff9999', linestyle='--', marker='o', label='Base (80m)'),
                Line2D([0], [0], color='#00008b', linestyle='-', marker='s', label='Ours (120m)'),
                Line2D([0], [0], color='#99ccff', linestyle='-', marker='s', label='Ours (80m)')
            ]
            ax3.legend(handles=custom_lines, loc='lower left', fontsize=8)

    # 总体排版
    plt.suptitle("Comprehensive Performance Disentanglement across Scenes and Factors", 
                 fontsize=18, weight='bold', y=0.92)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 顶级矩阵图表已生成: {save_path}")

if __name__ == "__main__":
    baseline_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    ours_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    output_pdf = "./Scene_Wise_Factor_Matrix.pdf"
    
    df = load_data(baseline_csv, ours_csv)
    plot_scene_wise_matrix(df, output_pdf)