

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import warnings

# warnings.filterwarnings("ignore")

# # ================= 1. 全局学术风格配置 (极简 & 极粗) =================
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'STIXGeneral'],
#     'mathtext.fontset': 'stix', 
#     'font.size': 24,              # 字号进一步微调
#     'font.weight': 'normal',        
#     'axes.labelweight': 'normal',   
#     'axes.titleweight': 'normal',   
#     'axes.linewidth': 2.5,        
#     'figure.dpi': 300,            
#     'savefig.bbox': 'tight',
# })

# METHODS = ['MoGe2', 'UniDepthV2', 'MoGe2-Aerial']

# # ================= 2. 数据读取 (保持逻辑一致) =================
# def load_all_data(baseline_csv, ours_csv, txt_ud):
#     df_base = pd.read_csv(baseline_csv)
#     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
#     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
#     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
#     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
#     df_txt = pd.read_csv(txt_ud, sep='\t')
#     df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
#     df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
#     df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
#     df_sub.rename(columns={'depth_metric.delta1': 'UniDepthV2'}, inplace=True)
#     df_merged = pd.merge(df_merged, df_sub, on=['Scene', 'Filename'], how='inner')
    
#     scene_map = {
#         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
#         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
#     }
#     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
#     for m in METHODS:
#         if df_merged[m].max() <= 1.05: df_merged[m] *= 100
#     return df_merged

# # ================= 3. 核心绘图函数 (无标题 & 全量热力版) =================
# # ================= 3. 核心绘图函数 (无标题 & 全量热力版) =================
# def plot_performance_matrices_premium(df, scene_name="Global"):
#     data_sub = df if scene_name == "Global" else df[df['Scene'] == scene_name]

#     # 1x3 布局，去掉顶部的标题空间
#     fig, axes = plt.subplots(1, 3, figsize=(26, 8.5), sharey=True)
#     plt.subplots_adjust(wspace=0.18) 
    
#     cmap = "RdYlGn" 
    
#     for i, m in enumerate(METHODS):
#         # 准备矩阵数据
#         matrix = data_sub.groupby(['Height_Group', 'Pitch_Group'])[m].mean().unstack()
#         matrix = matrix.sort_index(ascending=False)
#         matrix = matrix.reindex(columns=[-90, -75, -60, -45])

#         # 核心设置：0-100 全量色域
#         sns.heatmap(matrix, 
#                     annot=True, 
#                     fmt=".1f", 
#                     cmap=cmap, 
#                     vmin=0, vmax=100, 
#                     ax=axes[i], 
#                     cbar=False, 
#                     square=True, 
#                     linewidths=3.5, 
#                     linecolor='white',
#                     # 这里保持不变：只有数字加粗，大小维持24
#                     annot_kws={"size": 25, "weight": "bold"})
        
#         # 数值颜色自动对比优化
#         for text in axes[i].texts:
#             val = float(text.get_text())
#             text.set_text(f"{val:.1f}%")
#             if val < 30 or val > 80: # 极深色区域使用白字
#                 text.set_color("white")
#             else:
#                 text.set_color("black")

#         # 【修改1】子图标题：去掉加粗 (weight='normal')，字体从 36 缩小到 26
#         axes[i].set_title(m, fontsize=30, weight='normal', pad=15)
        
#         # 【修改2】X轴标签：去掉加粗，字体从 30 缩小到 22
#         axes[i].set_xlabel('Pitch Angle', fontsize=30, weight='normal', labelpad=15)
        
#         # 【修改3】X轴刻度：去掉加粗，字体从 26 缩小到 18
#         axes[i].set_xticklabels([f"{int(float(t.get_text()))}$^\circ$" for t in axes[i].get_xticklabels()], 
#                                 fontsize=30, weight='normal')

#         if i == 0:
#             # 【修改4】Y轴标签与刻度：去掉加粗，字体缩小
#             axes[i].set_ylabel('Altitude', fontsize=30, weight='normal', labelpad=15)
#             axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=18, weight='normal')
#         else:
#             axes[i].set_ylabel('')

#     # --- 独立 Colorbar 控制 ---
#     divider = make_axes_locatable(axes[2])
#     cax = divider.append_axes("right", size="5%", pad=0.3)
#     cbar = plt.colorbar(axes[2].get_children()[0], cax=cax)
#     cbar.outline.set_linewidth(2.5)
    
#     # 【修改5】Colorbar 标签：去掉加粗，字体从 26 缩小到 22
#     cbar.set_label('Accuracy (%)', fontsize=28, weight='normal', labelpad=15)
#     # 【修改6】Colorbar 刻度：字体从 22 缩小到 18
#     cbar.ax.tick_params(labelsize=22)

#     # ================= 变动在这里 =================
#     # 保存文件为 PDF 格式
#     filename = f"Final_Heatmap_NoTitle_{scene_name}.pdf"
#     plt.savefig(filename, format='pdf', bbox_inches='tight')
#     plt.close()
#     print(f"✅ 已生成无标题全量感热力图 (PDF): {filename}")

# # ================= 4. 执行 =================
# if __name__ == "__main__":
#     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
#     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
#     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

#     try:
#         df = load_all_data(baseline_csv, ours_csv, txt_ud)
#         # 仅绘制全局性能矩阵
#         plot_performance_matrices_premium(df, "Global")
#         # 如果需要分场景，也可以在这里运行：
#         # for scene in ['Campus', 'Factory', 'Farm', 'Grass']:
#         #     plot_performance_matrices_premium(df, scene)
#     except Exception as e:
#         print(f"运行时出错: {e}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 全局学术风格配置 (极简 & 极粗) =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral'],
    'mathtext.fontset': 'stix', 
    'font.size': 24,              # 字号进一步微调
    'font.weight': 'normal',        
    'axes.labelweight': 'normal',   
    'axes.titleweight': 'normal',   
    'axes.linewidth': 2.5,        
    'figure.dpi': 300,            
    'savefig.bbox': 'tight',
})

METHODS = ['MoGe2', 'UniDepthV2', 'MoGe2-Aerial']

# ================= 2. 数据读取 (保持逻辑一致) =================
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
    
    scene_map = {
        'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
        'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
    }
    df_merged['Scene'] = df_merged['Scene'].map(scene_map)
    for m in METHODS:
        if df_merged[m].max() <= 1.05: df_merged[m] *= 100
    return df_merged

# ================= 3. 核心绘图函数 (无标题 & 全量热力版) =================
def plot_performance_matrices_premium(df, scene_name="Global"):
    data_sub = df if scene_name == "Global" else df[df['Scene'] == scene_name]

    # 1x3 布局，去掉顶部的标题空间
    fig, axes = plt.subplots(1, 3, figsize=(26, 8.5), sharey=True)
    plt.subplots_adjust(wspace=0.18) 
    
    cmap = "RdYlGn" 
    
    for i, m in enumerate(METHODS):
        # 准备矩阵数据
        matrix = data_sub.groupby(['Height_Group', 'Pitch_Group'])[m].mean().unstack()
        matrix = matrix.sort_index(ascending=False)
        matrix = matrix.reindex(columns=[-90, -75, -60, -45])

        # 核心设置：0-100 全量色域
        sns.heatmap(matrix, 
                    annot=True, 
                    fmt=".1f", 
                    cmap=cmap, 
                    vmin=0, vmax=100, 
                    ax=axes[i], 
                    cbar=False, 
                    square=True, 
                    linewidths=3.5, 
                    linecolor='white',
                    # 这里保持不变：只有数字加粗，大小维持25
                    annot_kws={"size": 25, "weight": "bold"})
        
        # 【修改点】：强制所有数值使用黑色
        for text in axes[i].texts:
            val = float(text.get_text())
            text.set_text(f"{val:.1f}%")
            text.set_color("black")

        # 子图标题：去掉加粗 (weight='normal')
        axes[i].set_title(m, fontsize=30, weight='normal', pad=15)
        
        # X轴标签：去掉加粗
        axes[i].set_xlabel('Pitch Angle', fontsize=30, weight='normal', labelpad=15)
        
        # X轴刻度：去掉加粗
        axes[i].set_xticklabels([f"{int(float(t.get_text()))}$^\circ$" for t in axes[i].get_xticklabels()], 
                                fontsize=30, weight='normal')

        if i == 0:
            # Y轴标签与刻度：去掉加粗
            axes[i].set_ylabel('Altitude', fontsize=30, weight='normal', labelpad=15)
            axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=18, weight='normal')
        else:
            axes[i].set_ylabel('')

    # --- 独立 Colorbar 控制 ---
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cbar = plt.colorbar(axes[2].get_children()[0], cax=cax)
    cbar.outline.set_linewidth(2.5)
    
    # Colorbar 标签：去掉加粗
    cbar.set_label('Metric Accuracy ($\delta 1$) ', fontsize=28, weight='normal', labelpad=15)
    # Colorbar 刻度
    cbar.ax.tick_params(labelsize=22)

    # 保存文件为 PDF 格式
    filename = f"heatmap.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成无标题全量感热力图 (PDF): {filename}")

# ================= 4. 执行 =================
if __name__ == "__main__":
    baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

    try:
        df = load_all_data(baseline_csv, ours_csv, txt_ud)
        # 仅绘制全局性能矩阵
        plot_performance_matrices_premium(df, "Global")
        # 如果需要分场景，也可以在这里运行：
        # for scene in ['Campus', 'Factory', 'Farm', 'Grass']:
        #     plot_performance_matrices_premium(df, scene)
    except Exception as e:
        print(f"运行时出错: {e}")