

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# =====================================================================
# 🚀 顶会级学术绘图配置中心 (已适配 Altitude 标签)
# =====================================================================
CONFIG = {
    'font.family': 'serif',
    'axes.labelsize': 32,             
    'xtick.labelsize': 38,            
    'ytick.labelsize': 38,            
    'legend.fontsize': 30,            
    'value.fontsize': 30,             
    
    'bar.width': 0.16,                
    'bar.spacing': 0.03,              
    'colors': ['#D1EC86', '#60BA62', "#FFF5AE", "#FDB163"], 
    'edgecolor': 'black',             
    'edgewidth': 2.0,                 
    
    'axes.linewidth': 3.0,            
    'tick.major.width': 3.0,          
    'tick.major.size': 12,            
    'tick.direction': 'in',           
    
    'save_dpi': 600,                  
}

plt.rcParams.update({
    'font.family': CONFIG['font.family'],
    'font.serif': ['STIXGeneral', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'stix', 
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 300
})

METHODS = ['MoGe2', 'UniDepthV2', 'MoGe2-Aerial']

# ================= 数据读取函数 =================
def load_all_data(baseline_csv, ours_csv, txt_ud):
    df_base = pd.read_csv(baseline_csv)
    keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
    df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
    df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    df_txt = pd.read_csv(txt_ud, sep='\t')
    df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
    df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
    df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
    df_sub.rename(columns={'depth_metric.delta1': 'UniDepthV2'}, inplace=True)
    df_merged = pd.merge(df_merged, df_sub, on=['Scene', 'Filename'], how='inner')
    for m in METHODS:
        if df_merged[m].max() <= 1.05: df_merged[m] *= 100
    return df_merged

# ================= 🎯 修改点 1：将标签生成从 Height 改为 Altitude =================
def extract_plot_data(df):
    results = []
    # 1. 提取 FOV 数据
    df_fov = df.groupby('FOV_Group')[METHODS].mean().reset_index()
    for _, row in df_fov.iterrows():
        results.append({'Category': f"FOV {row['FOV_Group']}°", **{m: row[m] for m in METHODS}})
        
    # 2. 提取 Altitude 数据
    if 'Height_Group' in df.columns:
        df_h = df.groupby('Height_Group')[METHODS].mean().reset_index()
        for _, row in df_h.iterrows():
            if row['Height_Group'] in [80, 120]:
                # 此处字符串已修改：Height -> Altitude
                results.append({'Category': f"Altitude {row['Height_Group']}m", **{m: row[m] for m in METHODS}})
    return pd.DataFrame(results)

def save_dual_format(fig, base_name):
    fig.savefig(f"{base_name}.png", dpi=CONFIG['save_dpi'], bbox_inches='tight')
    fig.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    print(f"✅ 已保存: {base_name}.png & .pdf")

# ================= 核心绘图函数：优化布局与图例 =================
def plot_academic_bar_chart(df, base_filename="Method_Comparison_Academic"):
    df_plot = extract_plot_data(df)
    categories = df_plot['Category'].tolist()
    n_categories = len(categories)
    colors = CONFIG['colors']
    x = np.arange(len(METHODS))
    
    fig, ax = plt.subplots(figsize=(14, 7)) 
    
    total_width = CONFIG['bar.width'] * n_categories + CONFIG['bar.spacing'] * (n_categories - 1)
    start_offset = -total_width / 2 + CONFIG['bar.width'] / 2
    
    for i, cat in enumerate(categories):
        offset = start_offset + i * (CONFIG['bar.width'] + CONFIG['bar.spacing'])
        values = df_plot[METHODS].iloc[i].values
        bars = ax.bar(x + offset, values, width=CONFIG['bar.width'], label=cat, 
                      color=colors[i], edgecolor=CONFIG['edgecolor'], 
                      linewidth=CONFIG['edgewidth'], zorder=3)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 10), textcoords="offset points", ha='center', va='bottom',
                            fontsize=CONFIG['value.fontsize'], fontweight='normal')

    ax.set_ylabel(r'Metric Accuracy ($\delta 1$) %', fontsize=CONFIG['axes.labelsize'], fontweight='normal', labelpad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=CONFIG['xtick.labelsize'], fontweight='normal')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(CONFIG['ytick.labelsize'])
        label.set_fontweight('normal')
    
    # 🎯 修改点 2：压缩顶部空白 (从 120 降至 115，让画面更饱满)
    ax.set_ylim(0, 115) 
    
    for spine in ax.spines.values():
        spine.set_linewidth(CONFIG['axes.linewidth'])
    
    ax.tick_params(which='major', width=CONFIG['tick.major.width'], length=CONFIG['tick.major.size'], 
                   direction=CONFIG['tick.direction'], top=False, pad=15  )
    
    # 🎯 修改点 3：图例进一步下移 (从 0.92 降至 0.88)
    ax.legend(
        fontsize=CONFIG['legend.fontsize'], 
        loc='upper left', 
        bbox_to_anchor=(0.02, 0.88), # 数值 0.88 让图例距离顶端更远
        ncol=1, 
        frameon=True, 
        edgecolor='black', 
        fancybox=False, 
        shadow=False
    )
    
    plt.tight_layout()
    save_dual_format(fig, base_filename)
    plt.close()

# ================= 主运行入口 =================
if __name__ == "__main__":
    baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

    df = load_all_data(baseline_csv, ours_csv, txt_ud)
    plot_academic_bar_chart(df, "Academic_Altitude_Comparison")