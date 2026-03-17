


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# ================= 学术绘图全局设置 =================
plt.rcParams.update({
    'font.family': 'serif',
    # 优先使用内置的 STIXGeneral，它是最接近 Times New Roman 的自带字体
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix', # 让图表里的数学符号也保持 Times 风格
    
    # 统一调整为 18 号字体
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    
    # 标题稍微大一点点以示区分
    'axes.titlesize': 20, 
    'figure.dpi': 300
})

COLORS = {
    'MoGe2':        '#E64B35',   
    'DepthPro':     '#00A087',   
    'UniDepth':     '#3C5488',   
    'ZoeDepth':     '#F39B7F',   
    'MoGe2-Aerial': '#4DBBD5'    
}

COLOR_PITCH = '#196F3D'    
COLOR_HEIGHT = '#D35400'   
COLOR_FOV = '#922B21'      

def load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd):
    df_base = pd.read_csv(baseline_csv)
    keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    
    # 修改列重命名
    df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
    df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
    df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
    def load_txt(txt_path, col_name):
        df_txt = pd.read_csv(txt_path, sep='\t')
        df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
        df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
        df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
        df_sub.rename(columns={'depth_metric.delta1': col_name}, inplace=True)
        return df_sub

    df_depthpro = load_txt(txt_dp, 'DepthPro')
    df_unidepth = load_txt(txt_ud, 'UniDepth')
    df_zoedepth = load_txt(txt_zd, 'ZoeDepth')
    
    for df_method in [df_depthpro, df_unidepth, df_zoedepth]:
        df_merged = pd.merge(df_merged, df_method, on=['Scene', 'Filename'], how='inner')
        
    scene_map = {
        'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
        'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
    }
    df_merged['Scene'] = df_merged['Scene'].map(scene_map)
    
    # 修改方法列表
    methods = ['MoGe2', 'DepthPro', 'UniDepth', 'ZoeDepth', 'MoGe2-Aerial']
    for m in methods:
        if df_merged[m].max() <= 1.05:
            df_merged[m] *= 100
            
    return df_merged

# ================= 核心绘图逻辑 =================
def draw_radar_on_ax(ax, df_scene, scene_name, show_legend=False):
    factors = [
        'Pitch\n-90°', 'Pitch\n-75°', 'Pitch\n-60°', 'Pitch\n-45°', 
        'Altitude\n80m', 'Altitude\n120m', 
        'FOV\n63°', 'FOV\n83°'
    ]
    
    # 修改方法列表
    methods = ['MoGe2', 'DepthPro', 'UniDepth', 'ZoeDepth', 'MoGe2-Aerial']
    
    def get_method_vals(m_name):
        return [
            df_scene[df_scene['Pitch_Group'] == -90][m_name].mean(),
            df_scene[df_scene['Pitch_Group'] == -75][m_name].mean(),
            df_scene[df_scene['Pitch_Group'] == -60][m_name].mean(),
            df_scene[df_scene['Pitch_Group'] == -45][m_name].mean(),
            df_scene[df_scene['Height_Group'] == 80][m_name].mean(),
            df_scene[df_scene['Height_Group'] == 120][m_name].mean(),
            df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].min()][m_name].mean(),
            df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].max()][m_name].mean()
        ]
    
    vals_dict = {m: get_method_vals(m) for m in methods}
    angles = [n / float(len(factors)) * 2 * pi for n in range(len(factors))]
    
    ax.bar(x=3*pi/8, height=105, width=pi, color=COLOR_PITCH, alpha=0.08, zorder=0, edgecolor='none')
    ax.bar(x=9*pi/8, height=105, width=pi/2, color=COLOR_HEIGHT, alpha=0.08, zorder=0, edgecolor='none')
    ax.bar(x=13*pi/8, height=105, width=pi/2, color=COLOR_FOV, alpha=0.08, zorder=0, edgecolor='none')
    
    angles_closed = angles + angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    ax.spines['polar'].set_visible(False)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.tick_params(axis='x', pad=18)
    ax.set_xticks(angles)
    labels = ax.set_xticklabels(factors, weight='bold', size=16)
    
    for j, label in enumerate(labels):
        if j < 4: label.set_color(COLOR_PITCH)
        elif j < 6: label.set_color(COLOR_HEIGHT)
        else: label.set_color(COLOR_FOV)
    
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], color="grey", size=16)
    ax.set_rlabel_position(22.5)
    
    # ================= 🚀 核心逻辑：处理全0数据 =================
    zero_radii = [1.5, 3.5, 5.5] 
    zero_counter = 0
    
    for m in methods:
        current_vals = vals_dict[m]
        current_vals_closed = current_vals + current_vals[:1]
        
        is_all_zero = np.max(current_vals) < 1e-5
        
        if is_all_zero:
            r = zero_radii[zero_counter % len(zero_radii)]
            zero_vals_closed = [r] * len(angles_closed)
            zero_counter += 1
            
            ax.plot(angles_closed, zero_vals_closed, color=COLORS[m], 
                    linewidth=1.2, linestyle='--', zorder=4, alpha=0.8)
        else:
            # 修改高亮判断
            is_ours = (m == 'MoGe2-Aerial')
            lw = 2.5 if is_ours else 1.5
            alpha_fill = 0.25 if is_ours else 0.05
            zorder_line = 5 if is_ours else 3
            zorder_fill = 4 if is_ours else 2
            
            ax.plot(angles_closed, current_vals_closed, color=COLORS[m], 
                    linewidth=lw, linestyle='solid', zorder=zorder_line)
            ax.fill(angles_closed, current_vals_closed, color=COLORS[m], 
                    alpha=alpha_fill, zorder=zorder_fill)

    ax.set_title(f"[{scene_name}]", weight='bold', fontsize=16, pad=30, color='#333333')
    
    if show_legend:
        custom_lines = [
            Line2D([0], [0], color=COLORS[m], 
                   lw=2.5 if m=='MoGe2-Aerial' else 2, # 修改线宽判断
                   linestyle='--' if np.max(vals_dict[m]) < 1e-5 else '-',
                   label=m) 
            for m in methods
        ]
        ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=False, fontsize=18)

def plot_combined_radars(df, save_path_png):
    scenes = ['Campus', 'Factory', 'Farm', 'Grass']
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(22, 6), 
                             subplot_kw={'projection': 'polar'}, 
                             gridspec_kw={'wspace': 0.5})
    
    for i, scene in enumerate(scenes):
        df_scene = df[df['Scene'] == scene]
        draw_radar_on_ax(axes[i], df_scene, scene, show_legend=False)

    # 修改底部图例
    methods_list = ['MoGe2', 'DepthPro', 'UniDepth', 'ZoeDepth', 'MoGe2-Aerial']
    custom_lines = [
        Line2D([0], [0], color=COLORS[m], 
               lw=2.5 if m=='MoGe2-Aerial' else 2, 
               marker='s' if m=='MoGe2-Aerial' else 'o', 
               label=m) 
        for m in methods_list
    ]
    fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
               ncol=5, frameon=False, fontsize=23, columnspacing=3.0)

    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    save_path_pdf = save_path_png.replace('.png', '.pdf')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 联合雷达图已生成: {save_path_png} & {save_path_pdf}")

def plot_single_radars(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    scenes = ['Campus', 'Factory', 'Farm', 'Grass']
    
    for scene in scenes:
        df_scene = df[df['Scene'] == scene]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        draw_radar_on_ax(ax, df_scene, scene, show_legend=True)
        
        png_path = os.path.join(output_dir, f"Radar_{scene}.png")
        pdf_path = os.path.join(output_dir, f"Radar_{scene}.pdf")
        
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅ 独立雷达图源文件已生成: {png_path} & {pdf_path}")

if __name__ == "__main__":
    baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    
    txt_dp = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/ml_depth_pro_Bench_per_image.txt"
    txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"
    txt_zd = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/zoedepth_nk_metric_benchmarks_Bench_per_image.txt"

    df = load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd)
    plot_combined_radars(df, "Combined_Grouped_Radars_Horizontal_5_Methods.png")
    plot_single_radars(df, "Single_Radars")


# 1. 字体与字号调节 (Fonts & Sizes)
# 全局基础设置：搜索 plt.rcParams.update。这里控制免安装的学术字体（STIXGeneral）和全局基础字号（如 font.size、axes.labelsize）。

# 雷达图外圈标签（如 Pitch -90°）：搜索 ax.set_xticklabels。修改其中的 size 参数来放大或缩小外围文字。

# 子图标题（如 [Campus]）：搜索 ax.set_title。修改其中的 fontsize 控制大小，修改 pad 控制标题距离雷达图的高低。

# 图例字号：搜索 fig.legend（联合大图）或 ax.legend（单图）。修改其中的 fontsize。

# 2. 排版与图表间距 (Layout & Spacing)
# 长图总宽与总高：搜索 figsize。在 plot_combined_radars 中，如 figsize=(26, 6)。如果字号调大后边缘被裁切，继续加大宽度（如 28 或 30）。

# 子图水平间距（防止文字打架）：搜索 wspace。在 gridspec_kw={'wspace': 0.5} 中，数值越大，四个雷达图之间隔得越开。

# 外围文字与图形的距离：搜索 pad=（在 ax.tick_params 中）。如果字变大了觉得散，把 pad（比如从 18 降到 12）改小，文字就会向中心聚拢。

# 3. 颜色与视觉层级 (Colors & Aesthetics)
# 各方法的标准配色：搜索 COLORS = { 字典。可以随时替换为你偏好的 Nature-level 调色盘 HEX 值。

# 重点方法高亮控制：搜索 is_ours = (m == 'Ours')。这里的逻辑控制了主推方法的线宽（lw 变粗）、阴影透明度（alpha_fill 加深）和图层顺序（zorder 置顶），确保目标方法在视觉上最抢眼。

# 4. 特殊数据处理 (Data Handling)
# 全零指标的同心虚线圈：搜索 zero_radii。目前的默认值是 [1.5, 3.5, 5.5]，如果觉得原点附近的虚线圈太挤，可以直接拉大这几个数字。


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from math import pi
# import os
# import warnings
# from matplotlib.lines import Line2D
# warnings.filterwarnings("ignore")

# # =====================================================================
# # 🎛️ 学术绘图全局控制中心 (PLOT_CONFIG)
# # 所有需要微调的视觉参数都在这里，修改后直接生效
# # =====================================================================
# PLOT_CONFIG = {
#     # --- 1. 字体与字号 (Font & Sizes) ---
#     'font_family': ['STIXGeneral', 'DejaVu Serif', 'serif'], # 适配 Linux 的免安装 Times 风格
#     'base_fontsize': 14,       # 全局基础字号
#     'tick_label_size': 14,     # 外圈标签字号 (Pitch, FOV 等)
#     'inner_tick_size': 10,     # 雷达图内部网格刻度字号 (20, 40...)
#     'title_fontsize': 18,      # 子图标题字号 ([Campus] 等)
#     'legend_fontsize': 15,     # 图例字号

#     # --- 2. 排版与间距 (Layout & Spacing) ---
#     'combined_figsize': (28, 6), # 1x4 长图的总宽和高 (适当加宽，给大字号留足空间)
#     'single_figsize': (6, 6),    # 独立单图的宽高
#     'wspace': 0.5,               # 长图中子图的水平间距 (文字打架就调大此值)
#     'label_pad': 12,             # 外圈标签距离雷达图边缘的距离 (如果觉得散就改小)
#     'title_pad': 30,             # 标题距离顶部的距离

#     # --- 3. 颜色与视觉层级 (Colors & Aesthetics) ---
#     'method_colors': {
#         'Baseline': '#E64B35',   
#         'DepthPro': '#00A087',   
#         'UniDepth': '#3C5488',   
#         'ZoeDepth': '#F39B7F',   
#         'Ours':     '#4DBBD5'    
#     },
#     'bg_colors': {
#         'Pitch': '#196F3D',      # 深林绿
#         'Height': '#D35400',     # 焦糖橙
#         'FOV': '#922B21'         # 勃艮第深红
#     },
#     'ours_linewidth': 3.0,       # Ours 的线宽 (加粗以突出)
#     'other_linewidth': 1.5,      # 其他方法的线宽
#     'ours_fill_alpha': 0.25,     # Ours 的阴影透明度
#     'other_fill_alpha': 0.05,    # 其他方法的阴影透明度

#     # --- 4. 特殊数据处理 (Data Handling) ---
#     'zero_radii': [1.5, 3.5, 5.5], # 全0指标在中心生成的虚线圈半径大小 (觉得挤就调大)
#     'zero_threshold': 1e-5         # 判定为全0的极限阈值
# }
# # =====================================================================

# # 应用全局字体和字号设置
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': PLOT_CONFIG['font_family'],
#     'mathtext.fontset': 'stix',
#     'font.size': PLOT_CONFIG['base_fontsize'],
#     'axes.labelsize': PLOT_CONFIG['base_fontsize'],
#     'xtick.labelsize': PLOT_CONFIG['base_fontsize'],
#     'ytick.labelsize': PLOT_CONFIG['base_fontsize'],
#     'legend.fontsize': PLOT_CONFIG['legend_fontsize'],
#     'figure.dpi': 300
# })

# def load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd):
#     df_base = pd.read_csv(baseline_csv)
#     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    
#     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'Baseline'})
#     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'Ours'})
#     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
#     def load_txt(txt_path, col_name):
#         df_txt = pd.read_csv(txt_path, sep='\t')
#         df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
#         df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
#         df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
#         df_sub.rename(columns={'depth_metric.delta1': col_name}, inplace=True)
#         return df_sub

#     df_depthpro = load_txt(txt_dp, 'DepthPro')
#     df_unidepth = load_txt(txt_ud, 'UniDepth')
#     df_zoedepth = load_txt(txt_zd, 'ZoeDepth')
    
#     for df_method in [df_depthpro, df_unidepth, df_zoedepth]:
#         df_merged = pd.merge(df_merged, df_method, on=['Scene', 'Filename'], how='inner')
        
#     scene_map = {
#         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
#         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
#     }
#     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
    
#     methods = ['Baseline', 'DepthPro', 'UniDepth', 'ZoeDepth', 'Ours']
#     for m in methods:
#         if df_merged[m].max() <= 1.05:
#             df_merged[m] *= 100
            
#     return df_merged

# # ================= 核心绘图逻辑 =================
# def draw_radar_on_ax(ax, df_scene, scene_name, show_legend=False):
#     factors = [
#         'Pitch\n-90°', 'Pitch\n-75°', 'Pitch\n-60°', 'Pitch\n-45°', 
#         'Altitude\n80m', 'Altitude\n120m', 
#         'FOV\n63°', 'FOV\n83°'
#     ]
#     methods = list(PLOT_CONFIG['method_colors'].keys())
    
#     def get_method_vals(m_name):
#         return [
#             df_scene[df_scene['Pitch_Group'] == -90][m_name].mean(),
#             df_scene[df_scene['Pitch_Group'] == -75][m_name].mean(),
#             df_scene[df_scene['Pitch_Group'] == -60][m_name].mean(),
#             df_scene[df_scene['Pitch_Group'] == -45][m_name].mean(),
#             df_scene[df_scene['Height_Group'] == 80][m_name].mean(),
#             df_scene[df_scene['Height_Group'] == 120][m_name].mean(),
#             df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].min()][m_name].mean(),
#             df_scene[df_scene['FOV_Group'] == df_scene['FOV_Group'].max()][m_name].mean()
#         ]
    
#     vals_dict = {m: get_method_vals(m) for m in methods}
#     angles = [n / float(len(factors)) * 2 * pi for n in range(len(factors))]
    
#     # 扇形背景划分
#     ax.bar(x=3*pi/8, height=105, width=pi, color=PLOT_CONFIG['bg_colors']['Pitch'], alpha=0.08, zorder=0, edgecolor='none')
#     ax.bar(x=9*pi/8, height=105, width=pi/2, color=PLOT_CONFIG['bg_colors']['Height'], alpha=0.08, zorder=0, edgecolor='none')
#     ax.bar(x=13*pi/8, height=105, width=pi/2, color=PLOT_CONFIG['bg_colors']['FOV'], alpha=0.08, zorder=0, edgecolor='none')
    
#     angles_closed = angles + angles[:1]
    
#     ax.set_theta_offset(pi / 2)
#     ax.set_theta_direction(-1)
    
#     ax.spines['polar'].set_visible(False)
#     ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

#     ax.tick_params(axis='x', pad=PLOT_CONFIG['label_pad'])
#     ax.set_xticks(angles)
#     labels = ax.set_xticklabels(factors, weight='bold', size=PLOT_CONFIG['tick_label_size'])
    
#     for j, label in enumerate(labels):
#         if j < 4: label.set_color(PLOT_CONFIG['bg_colors']['Pitch'])
#         elif j < 6: label.set_color(PLOT_CONFIG['bg_colors']['Height'])
#         else: label.set_color(PLOT_CONFIG['bg_colors']['FOV'])
    
#     ax.set_ylim(0, 105)
#     ax.set_yticks([20, 40, 60, 80])
#     ax.set_yticklabels(['20', '40', '60', '80'], color="grey", size=PLOT_CONFIG['inner_tick_size'])
#     ax.set_rlabel_position(22.5)
    
#     # 处理全0数据与正常数据绘制
#     zero_radii = PLOT_CONFIG['zero_radii']
#     zero_counter = 0
    
#     for m in methods:
#         current_vals = vals_dict[m]
#         current_vals_closed = current_vals + current_vals[:1]
        
#         is_all_zero = np.max(current_vals) < PLOT_CONFIG['zero_threshold']
        
#         if is_all_zero:
#             r = zero_radii[zero_counter % len(zero_radii)]
#             zero_vals_closed = [r] * len(angles_closed)
#             zero_counter += 1
            
#             ax.plot(angles_closed, zero_vals_closed, color=PLOT_CONFIG['method_colors'][m], 
#                     linewidth=1.5, linestyle='--', zorder=4, alpha=0.8)
#         else:
#             is_ours = (m == 'Ours')
#             lw = PLOT_CONFIG['ours_linewidth'] if is_ours else PLOT_CONFIG['other_linewidth']
#             alpha_fill = PLOT_CONFIG['ours_fill_alpha'] if is_ours else PLOT_CONFIG['other_fill_alpha']
#             zorder_line = 5 if is_ours else 3
#             zorder_fill = 4 if is_ours else 2
            
#             ax.plot(angles_closed, current_vals_closed, color=PLOT_CONFIG['method_colors'][m], 
#                     linewidth=lw, linestyle='solid', zorder=zorder_line)
#             ax.fill(angles_closed, current_vals_closed, color=PLOT_CONFIG['method_colors'][m], 
#                     alpha=alpha_fill, zorder=zorder_fill)

#     ax.set_title(f"[{scene_name}]", weight='bold', fontsize=PLOT_CONFIG['title_fontsize'], 
#                  pad=PLOT_CONFIG['title_pad'], color='#333333')
    
#     if show_legend:
#         custom_lines = [
#             Line2D([0], [0], color=PLOT_CONFIG['method_colors'][m], 
#                    lw=PLOT_CONFIG['ours_linewidth'] if m=='Ours' else PLOT_CONFIG['other_linewidth'], 
#                    linestyle='--' if np.max(vals_dict[m]) < PLOT_CONFIG['zero_threshold'] else '-',
#                    label=m) 
#             for m in methods
#         ]
#         ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.35, 1.15), 
#                   frameon=False, fontsize=PLOT_CONFIG['legend_fontsize'])

# def plot_combined_radars(df, save_path_png):
#     scenes = ['Campus', 'Factory', 'Farm', 'Grass']
#     methods = list(PLOT_CONFIG['method_colors'].keys())
    
#     fig, axes = plt.subplots(nrows=1, ncols=4, figsize=PLOT_CONFIG['combined_figsize'], 
#                              subplot_kw={'projection': 'polar'}, 
#                              gridspec_kw={'wspace': PLOT_CONFIG['wspace']})
    
#     for i, scene in enumerate(scenes):
#         df_scene = df[df['Scene'] == scene]
#         draw_radar_on_ax(axes[i], df_scene, scene, show_legend=False)

#     custom_lines = [
#         Line2D([0], [0], color=PLOT_CONFIG['method_colors'][m], 
#                lw=PLOT_CONFIG['ours_linewidth'] if m=='Ours' else PLOT_CONFIG['other_linewidth'], 
#                marker='s' if m=='Ours' else 'o', label=m) 
#         for m in methods
#     ]
#     fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
#                ncol=len(methods), frameon=False, fontsize=PLOT_CONFIG['legend_fontsize'], columnspacing=3.0)

#     plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
#     save_path_pdf = save_path_png.replace('.png', '.pdf')
#     plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
#     plt.close()
#     print(f"✅ 联合雷达大图已生成: {save_path_png} & {save_path_pdf}")

# def plot_single_radars(df, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     scenes = ['Campus', 'Factory', 'Farm', 'Grass']
    
#     for scene in scenes:
#         df_scene = df[df['Scene'] == scene]
#         fig, ax = plt.subplots(figsize=PLOT_CONFIG['single_figsize'], subplot_kw={'projection': 'polar'})
        
#         draw_radar_on_ax(ax, df_scene, scene, show_legend=True)
        
#         png_path = os.path.join(output_dir, f"Radar_{scene}.png")
#         pdf_path = os.path.join(output_dir, f"Radar_{scene}.pdf")
        
#         plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
#         plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
#         plt.close()
#         print(f"✅ 独立雷达图源文件已生成: {png_path} & {pdf_path}")

# if __name__ == "__main__":
#     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
#     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    
#     txt_dp = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/ml_depth_pro_Bench_per_image.txt"
#     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"
#     txt_zd = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/zoedepth_nk_metric_benchmarks_Bench_per_image.txt"

#     df = load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd)
#     plot_combined_radars(df, "Combined_Grouped_Radars_Horizontal_5_Methods.png")
#     plot_single_radars(df, "Single_Radars")