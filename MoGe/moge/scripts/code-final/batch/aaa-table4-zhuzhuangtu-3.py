# # # # import pandas as pd
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # import seaborn as sns
# # # # from matplotlib.lines import Line2D
# # # # import warnings
# # # # warnings.filterwarnings("ignore")

# # # # # ================= 学术绘图全局设置 =================
# # # # plt.rcParams.update({
# # # #     'font.family': 'serif',
# # # #     'font.serif': ['Times New Roman', 'DejaVu Serif'],
# # # #     'axes.titlesize': 13,
# # # #     'axes.labelsize': 11,
# # # #     'xtick.labelsize': 10,
# # # #     'ytick.labelsize': 10,
# # # #     'figure.dpi': 300
# # # # })

# # # # def load_data(baseline_csv, ours_csv):
# # # #     df_base = pd.read_csv(baseline_csv)
# # # #     df_ours = pd.read_csv(ours_csv)
# # # #     scene_map = {'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# # # #                  'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'}
# # # #     df_base['Scene'] = df_base['Scene'].map(scene_map)
# # # #     df_ours['Scene'] = df_ours['Scene'].map(scene_map)
# # # #     keys = ['Filename', 'Scene', 'FOV_Group', 'Height_Group', 'Pitch_Group']
# # # #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Base'})
# # # #     df_o = df_ours[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Ours'})
# # # #     df = pd.merge(df_b, df_o, on=keys, how='inner')
# # # #     if df['a1.25_Base'].max() <= 1.05:
# # # #         df['a1.25_Base'] *= 100; df['a1.25_Ours'] *= 100
# # # #     return df

# # # # def plot_lineplots_only(df, save_path):
# # # #     scenes = ['Campus', 'Factory', 'Farm', 'Grass']
    
# # # #     fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 14), gridspec_kw={'hspace': 0.35})
    
# # # #     for i, scene in enumerate(scenes):
# # # #         ax = axes[i]
# # # #         df_scene = df[df['Scene'] == scene]
        
# # # #         sns.lineplot(data=df_scene, x='Pitch_Group', y='a1.25_Base', hue='Height_Group', 
# # # #                      palette={80: '#ff9999', 120: '#8b0000'}, linestyle='--', marker='o', ax=ax, legend=False)
# # # #         sns.lineplot(data=df_scene, x='Pitch_Group', y='a1.25_Ours', hue='Height_Group', 
# # # #                      palette={80: '#99ccff', 120: '#00008b'}, linestyle='-', marker='s', ax=ax, legend=False)
                     
# # # #         ax.set_title(f"[{scene}] Pitch & Height Disentanglement", weight='bold')
# # # #         ax.set_ylabel(r"Accuracy $a_{1.25}$ (%)")
        
# # # #         if i == 3:
# # # #             ax.set_xlabel("Camera Pitch Angle (°)")
# # # #         else:
# # # #             ax.set_xlabel("")
            
# # # #         ax.set_xticks([-90, -75, -60, -45])
# # # #         ax.set_ylim(-5, 105)
# # # #         ax.grid(True, linestyle=':', alpha=0.6)
        
# # # #         if i == 0:
# # # #             custom_lines = [
# # # #                 Line2D([0], [0], color='#8b0000', linestyle='--', marker='o', label='Base (120m)'),
# # # #                 Line2D([0], [0], color='#ff9999', linestyle='--', marker='o', label='Base (80m)'),
# # # #                 Line2D([0], [0], color='#00008b', linestyle='-', marker='s', label='Ours (120m)'),
# # # #                 Line2D([0], [0], color='#99ccff', linestyle='-', marker='s', label='Ours (80m)')
# # # #             ]
# # # #             ax.legend(handles=custom_lines, loc='lower left', fontsize=9)

# # # #     plt.suptitle("Pitch & Height Disentanglement across Scenes", fontsize=16, weight='bold', y=0.93)
# # # #     # 修改为保存 PNG 图片
# # # #     plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
# # # #     plt.close()
# # # #     print(f"✅ 交叉分析折线图已生成: {save_path}")

# # # # if __name__ == "__main__":
# # # #     baseline_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# # # #     ours_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# # # #     # 保存在当前目录的 png 文件
# # # #     output_png = "3_Lineplots_Only.png"
    
# # # #     df = load_data(baseline_csv, ours_csv)
# # # #     plot_lineplots_only(df, output_png)
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # from matplotlib.lines import Line2D
# # # import warnings
# # # warnings.filterwarnings("ignore")

# # # # ================= 学术绘图全局设置 =================
# # # plt.rcParams.update({
# # #     'font.family': 'serif',
# # #     'font.serif': ['Times New Roman', 'DejaVu Serif'],
# # #     'axes.titlesize': 13,
# # #     'axes.labelsize': 12,
# # #     'xtick.labelsize': 11,
# # #     'ytick.labelsize': 11,
# # #     'figure.dpi': 300,
# # #     'axes.linewidth': 1.0,
# # #     'axes.edgecolor': '#333333'
# # # })

# # # def load_data(baseline_csv, ours_csv):
# # #     df_base = pd.read_csv(baseline_csv)
# # #     df_ours = pd.read_csv(ours_csv)
# # #     scene_map = {'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# # #                  'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'}
# # #     df_base['Scene'] = df_base['Scene'].map(scene_map)
# # #     df_ours['Scene'] = df_ours['Scene'].map(scene_map)
# # #     keys = ['Filename', 'Scene', 'FOV_Group', 'Height_Group', 'Pitch_Group']
# # #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Base'})
# # #     df_o = df_ours[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Ours'})
# # #     df = pd.merge(df_b, df_o, on=keys, how='inner')
# # #     if df['a1.25_Base'].max() <= 1.05:
# # #         df['a1.25_Base'] *= 100; df['a1.25_Ours'] *= 100
# # #     return df

# # # def plot_lineplots_fov_grid(df, save_path):
# # #     scenes = ['Campus', 'Factory', 'Farm', 'Grass']
# # #     fovs = [df['FOV_Group'].min(), df['FOV_Group'].max()]
    
# # #     # 【顶级排版核心】：4行2列，共享 XY 轴，拉近间距，形成严密的网格对比
# # #     fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 11), 
# # #                              sharex=True, sharey=True, 
# # #                              gridspec_kw={'wspace': 0.08, 'hspace': 0.12})
    
# # #     # 颜色定义 (保持了你的原本选择，增加了透明度以便置信区间显示更好看)
# # #     color_base_80 = '#ff9999'
# # #     color_base_120 = '#8b0000'
# # #     color_ours_80 = '#99ccff'
# # #     color_ours_120 = '#00008b'

# # #     for i, scene in enumerate(scenes):
# # #         for j, fov in enumerate(fovs):
# # #             ax = axes[i, j]
# # #             df_sub = df[(df['Scene'] == scene) & (df['FOV_Group'] == fov)]
            
# # #             # 画 Baseline 折线 (带 95% 置信区间阴影)
# # #             sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Base', hue='Height_Group', 
# # #                          palette={80: color_base_80, 120: color_base_120}, 
# # #                          linestyle='--', marker='o', errorbar=('ci', 95), 
# # #                          linewidth=1.5, markersize=6, ax=ax, legend=False)
                         
# # #             # 画 Ours 折线 (带 95% 置信区间阴影)
# # #             sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Ours', hue='Height_Group', 
# # #                          palette={80: color_ours_80, 120: color_ours_120}, 
# # #                          linestyle='-', marker='s', errorbar=('ci', 95), 
# # #                          linewidth=2.0, markersize=6, ax=ax, legend=False)
            
# # #             # --- 网格与刻度 ---
# # #             ax.set_xticks([-90, -75, -60, -45])
# # #             ax.set_ylim(-5, 105)
# # #             ax.grid(True, linestyle='--', alpha=0.5, color='#999999')
            
# # #             # --- 去边框 (呼吸感) ---
# # #             sns.despine(ax=ax)
            
# # #             # --- 清理重复标签 ---
# # #             ax.set_title("")
# # #             ax.set_ylabel("")
# # #             ax.set_xlabel("")
            
# # #             # 【矩阵标签】列标题 (FOV)
# # #             if i == 0:
# # #                 ax.set_title(f"FOV = {fov}°", weight='bold', fontsize=13, pad=12)
            
# # #             # 【矩阵标签】行标题 (Scene)
# # #             if j == 1:
# # #                 ax.text(1.05, 0.5, scene, transform=ax.transAxes, 
# # #                         rotation=-90, va='center', ha='left', 
# # #                         weight='bold', fontsize=13, color='#333333')

# # #     # 全局坐标轴大标签
# # #     fig.supylabel(r"Accuracy $a_{1.25}$ (%)", fontsize=14, weight='bold', x=0.04, color='#333333')
# # #     fig.supxlabel("Camera Pitch Angle (°)", fontsize=14, weight='bold', y=0.04, color='#333333')

# # #     # 底部统一的极简图例
# # #     custom_lines = [
# # #         Line2D([0], [0], color=color_base_80, linestyle='--', marker='o', markersize=7, label='Base (80m)'),
# # #         Line2D([0], [0], color=color_base_120, linestyle='--', marker='o', markersize=7, label='Base (120m)'),
# # #         Line2D([0], [0], color=color_ours_80, linestyle='-', marker='s', markersize=7, label='Ours (80m)'),
# # #         Line2D([0], [0], color=color_ours_120, linestyle='-', marker='s', markersize=7, label='Ours (120m)')
# # #     ]
# # #     fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
# # #                ncol=4, frameon=False, fontsize=12, handletextpad=0.5, columnspacing=2.5)

# # #     # 保存
# # #     plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
# # #     plt.close()
# # #     print(f"✅ 包含 FOV 的顶级网格折线图已生成: {save_path}")

# # # if __name__ == "__main__":
# # #     baseline_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# # #     ours_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# # #     output_png = "3_Lineplots_Grid_FOV.png"
    
# # #     df = load_data(baseline_csv, ours_csv)
# # #     plot_lineplots_fov_grid(df, output_png)

# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from matplotlib.lines import Line2D
# # import warnings
# # warnings.filterwarnings("ignore")

# # # ================= 学术绘图全局设置 =================
# # plt.rcParams.update({
# #     'font.family': 'serif',
# #     'font.serif': ['Times New Roman', 'DejaVu Serif'],
# #     'axes.titlesize': 13,
# #     'axes.labelsize': 12,
# #     'xtick.labelsize': 11,
# #     'ytick.labelsize': 11,
# #     'figure.dpi': 300,
# #     'axes.linewidth': 1.0,
# #     'axes.edgecolor': '#333333'
# # })

# # def load_data(baseline_csv, ours_csv):
# #     df_base = pd.read_csv(baseline_csv)
# #     df_ours = pd.read_csv(ours_csv)
# #     scene_map = {'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# #                  'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'}
# #     df_base['Scene'] = df_base['Scene'].map(scene_map)
# #     df_ours['Scene'] = df_ours['Scene'].map(scene_map)
# #     keys = ['Filename', 'Scene', 'FOV_Group', 'Height_Group', 'Pitch_Group']
# #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Base'})
# #     df_o = df_ours[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Ours'})
# #     df = pd.merge(df_b, df_o, on=keys, how='inner')
# #     if df['a1.25_Base'].max() <= 1.05:
# #         df['a1.25_Base'] *= 100; df['a1.25_Ours'] *= 100
# #     return df

# # def plot_lineplots_fov_grid_clean(df, save_path):
# #     scenes = ['Campus', 'Factory', 'Farm', 'Grass']
# #     fovs = [df['FOV_Group'].min(), df['FOV_Group'].max()]
    
# #     # 稍微拉开一点 wspace，恢复你喜欢的留白与呼吸感
# #     fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 12), 
# #                              sharex=True, sharey=True, 
# #                              gridspec_kw={'wspace': 0.15, 'hspace': 0.25})
    
# #     color_base_80 = '#ff9999'
# #     color_base_120 = '#8b0000'
# #     color_ours_80 = '#99ccff'
# #     color_ours_120 = '#00008b'

# #     for i, scene in enumerate(scenes):
# #         for j, fov in enumerate(fovs):
# #             ax = axes[i, j]
# #             df_sub = df[(df['Scene'] == scene) & (df['FOV_Group'] == fov)]
            
# #             # 【核心修改】：errorbar=None，去掉所有多余的边缘阴影，恢复线条的锐利与纯粹！
# #             sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Base', hue='Height_Group', 
# #                          palette={80: color_base_80, 120: color_base_120}, 
# #                          linestyle='--', marker='o', errorbar=None, 
# #                          linewidth=1.5, markersize=7, ax=ax, legend=False)
                         
# #             sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Ours', hue='Height_Group', 
# #                          palette={80: color_ours_80, 120: color_ours_120}, 
# #                          linestyle='-', marker='s', errorbar=None, 
# #                          linewidth=2.0, markersize=7, ax=ax, legend=False)
            
# #             # --- 极简网格与坐标轴 ---
# #             ax.set_xticks([-90, -75, -60, -45])
# #             ax.set_ylim(-5, 105)
# #             # 内部保留极其柔和的虚线网格，辅助看分数
# #             ax.grid(True, linestyle=':', alpha=0.6, color='#888888')
            
# #             # 彻底去掉上下左右不必要的实线边框
# #             sns.despine(ax=ax)
            
# #             ax.set_title("")
# #             ax.set_ylabel("")
# #             ax.set_xlabel("")
            
# #             # 矩阵式标签：列标题
# #             if i == 0:
# #                 ax.set_title(f"FOV = {fov}°", weight='bold', fontsize=13, pad=15)
            
# #             # 矩阵式标签：行标题
# #             if j == 1:
# #                 ax.text(1.05, 0.5, f"[{scene}]", transform=ax.transAxes, 
# #                         rotation=-90, va='center', ha='left', 
# #                         weight='bold', fontsize=13, color='#333333')

# #     # 全局 XY 标签
# #     fig.supylabel(r"Accuracy $a_{1.25}$ (%)", fontsize=14, weight='bold', x=0.04, color='#333333')
# #     fig.supxlabel("Camera Pitch Angle (°)", fontsize=14, weight='bold', y=0.05, color='#333333')

# #     # 底部独立图例
# #     custom_lines = [
# #         Line2D([0], [0], color=color_base_120, linestyle='--', marker='o', markersize=8, label='Baseline (120m)'),
# #         Line2D([0], [0], color=color_base_80, linestyle='--', marker='o', markersize=8, label='Baseline (80m)'),
# #         Line2D([0], [0], color=color_ours_120, linestyle='-', marker='s', markersize=8, label='Ours (120m)'),
# #         Line2D([0], [0], color=color_ours_80, linestyle='-', marker='s', markersize=8, label='Ours (80m)')
# #     ]
# #     # 图例居中并排
# #     fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
# #                ncol=4, frameon=False, fontsize=12, handletextpad=0.5, columnspacing=2.5)

# #     plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
# #     plt.close()
# #     print(f"✅ 纯净版 FOV 对比折线图已生成: {save_path}")

# # if __name__ == "__main__":
# #     baseline_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# #     ours_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# #     output_png = "3_Lineplots_Grid_Clean.png"
    
# #     df = load_data(baseline_csv, ours_csv)
# #     plot_lineplots_fov_grid_clean(df, output_png)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D
# import warnings
# warnings.filterwarnings("ignore")

# # ================= 学术绘图全局设置 =================
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'DejaVu Serif'],
#     'axes.titlesize': 13,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 11,
#     'figure.dpi': 300,
#     'axes.linewidth': 1.0,
#     'axes.edgecolor': '#333333'
# })

# # ================= 顶级冷暖调色盘 =================
# COLOR_BASE_80 = '#D98880'   # 赤土陶粉
# COLOR_BASE_120 = '#7B241C'  # 勃艮第酒红
# COLOR_OURS_80 = '#7393B3'   # 灰霾蓝
# COLOR_OURS_120 = '#0F4C81'  # 深邃蓝宝石

# def load_data(baseline_csv, ours_csv):
#     df_base = pd.read_csv(baseline_csv)
#     df_ours = pd.read_csv(ours_csv)
#     scene_map = {'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
#                  'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'}
#     df_base['Scene'] = df_base['Scene'].map(scene_map)
#     df_ours['Scene'] = df_ours['Scene'].map(scene_map)
#     keys = ['Filename', 'Scene', 'FOV_Group', 'Height_Group', 'Pitch_Group']
#     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Base'})
#     df_o = df_ours[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Ours'})
#     df = pd.merge(df_b, df_o, on=keys, how='inner')
#     if df['a1.25_Base'].max() <= 1.05:
#         df['a1.25_Base'] *= 100; df['a1.25_Ours'] *= 100
#     return df

# def plot_lineplots_fov_grid_ci(df, save_path):
#     scenes = ['Campus', 'Factory', 'Farm', 'Grass']
#     fovs = [df['FOV_Group'].min(), df['FOV_Group'].max()]
    
#     fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 12), 
#                              sharex=True, sharey=True, 
#                              gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
                             
#     # 统一的阴影参数：去除边缘线，设置 15% 极致透明度
#     ci_kws = {'edgecolor': 'none', 'alpha': 0.15}

#     for i, scene in enumerate(scenes):
#         for j, fov in enumerate(fovs):
#             ax = axes[i, j]
#             df_sub = df[(df['Scene'] == scene) & (df['FOV_Group'] == fov)]
            
#             # 【Baseline】：带有无边框透明阴影的折线
#             sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Base', hue='Height_Group', 
#                          palette={80: COLOR_BASE_80, 120: COLOR_BASE_120}, 
#                          linestyle='--', marker='o', errorbar=('ci', 95), 
#                          err_kws=ci_kws, # 这里的魔法参数去掉了阴影边缘
#                          linewidth=1.5, markersize=6, ax=ax, legend=False)
                         
#             # 【Ours】：带有无边框透明阴影的折线
#             sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Ours', hue='Height_Group', 
#                          palette={80: COLOR_OURS_80, 120: COLOR_OURS_120}, 
#                          linestyle='-', marker='s', errorbar=('ci', 95), 
#                          err_kws=ci_kws, # 这里的魔法参数去掉了阴影边缘
#                          linewidth=2.0, markersize=6, ax=ax, legend=False)
            
#             # --- 极简网格 ---
#             ax.set_xticks([-90, -75, -60, -45])
#             ax.set_ylim(-5, 105)
#             ax.grid(True, linestyle=':', alpha=0.5, color='#999999')
#             sns.despine(ax=ax)
            
#             ax.set_title("")
#             ax.set_ylabel("")
#             ax.set_xlabel("")
            
#             # 列标题
#             if i == 0:
#                 ax.set_title(f"FOV = {fov}°", weight='bold', fontsize=13, pad=15)
            
#             # 行标题 (场景名)
#             if j == 1:
#                 ax.text(1.05, 0.5, f"[{scene}]", transform=ax.transAxes, 
#                         rotation=-90, va='center', ha='left', 
#                         weight='bold', fontsize=13, color='#333333')

#     # 全局坐标轴标签
#     fig.supylabel(r"Accuracy $a_{1.25}$ (%)", fontsize=14, weight='bold', x=0.04, color='#333333')
#     fig.supxlabel("Camera Pitch Angle (°)", fontsize=14, weight='bold', y=0.04, color='#333333')

#     # 图例
#     custom_lines = [
#         Line2D([0], [0], color=COLOR_BASE_120, linestyle='--', marker='o', markersize=7, label='Baseline (120m)'),
#         Line2D([0], [0], color=COLOR_BASE_80, linestyle='--', marker='o', markersize=7, label='Baseline (80m)'),
#         Line2D([0], [0], color=COLOR_OURS_120, linestyle='-', marker='s', markersize=7, label='Ours (120m)'),
#         Line2D([0], [0], color=COLOR_OURS_80, linestyle='-', marker='s', markersize=7, label='Ours (80m)')
#     ]
#     fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.01), 
#                ncol=4, frameon=False, fontsize=12, handletextpad=0.5, columnspacing=2.5)

#     plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✅ 完美清透置信区间折线图已生成: {save_path}")

# if __name__ == "__main__":
#     baseline_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
#     ours_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
#     output_png = "3_Lineplots_Grid_Perfect_CI.png"
    
#     df = load_data(baseline_csv, ours_csv)
#     plot_lineplots_fov_grid_ci(df, output_png)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ================= 学术绘图全局设置 =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333'
})

# ================= 顶级冷暖调色盘 =================
COLOR_BASE_80 = '#D98880'   # 赤土陶粉
COLOR_BASE_120 = '#7B241C'  # 勃艮第酒红
COLOR_OURS_80 = '#7393B3'   # 灰霾蓝
COLOR_OURS_120 = '#0F4C81'  # 深邃蓝宝石

def load_data(baseline_csv, ours_csv):
    df_base = pd.read_csv(baseline_csv)
    df_ours = pd.read_csv(ours_csv)
    scene_map = {'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
                 'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'}
    df_base['Scene'] = df_base['Scene'].map(scene_map)
    df_ours['Scene'] = df_ours['Scene'].map(scene_map)
    keys = ['Filename', 'Scene', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Base'})
    df_o = df_ours[keys + ['a1.25']].rename(columns={'a1.25': 'a1.25_Ours'})
    df = pd.merge(df_b, df_o, on=keys, how='inner')
    if df['a1.25_Base'].max() <= 1.05:
        df['a1.25_Base'] *= 100; df['a1.25_Ours'] *= 100
    return df

def plot_lineplots_horizontal_grid(df, save_path):
    scenes = ['Campus', 'Factory', 'Farm', 'Grass']
    fovs = [df['FOV_Group'].min(), df['FOV_Group'].max()]
    
    # 【核心改动】：2行4列，横向宽幅画卷 (15x6 比例极佳)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 6), 
                             sharex=True, sharey=True, 
                             gridspec_kw={'wspace': 0.08, 'hspace': 0.15})
                             
    # 统一的阴影参数：无边缘，15% 极致透明度
    ci_kws = {'edgecolor': 'none', 'alpha': 0.15}

    # 外层循环改为 FOV (行)，内层循环改为 Scene (列)
    for i, fov in enumerate(fovs):
        for j, scene in enumerate(scenes):
            ax = axes[i, j]
            df_sub = df[(df['Scene'] == scene) & (df['FOV_Group'] == fov)]
            
            # 【Baseline】
            sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Base', hue='Height_Group', 
                         palette={80: COLOR_BASE_80, 120: COLOR_BASE_120}, 
                         linestyle='--', marker='o', errorbar=('ci', 95), 
                         err_kws=ci_kws, linewidth=1.5, markersize=6, ax=ax, legend=False)
                         
            # 【Ours】
            sns.lineplot(data=df_sub, x='Pitch_Group', y='a1.25_Ours', hue='Height_Group', 
                         palette={80: COLOR_OURS_80, 120: COLOR_OURS_120}, 
                         linestyle='-', marker='s', errorbar=('ci', 95), 
                         err_kws=ci_kws, linewidth=2.0, markersize=6, ax=ax, legend=False)
            
            # --- 极简网格 ---
            ax.set_xticks([-90, -75, -60, -45])
            ax.set_ylim(-5, 105)
            ax.grid(True, linestyle=':', alpha=0.5, color='#999999')
            sns.despine(ax=ax)
            
            ax.set_title("")
            ax.set_ylabel("")
            ax.set_xlabel("")
            
            # 【矩阵标签】顶部的列标题 (Scene)
            if i == 0:
                ax.set_title(f"[{scene}]", weight='bold', fontsize=14, pad=12)
            
            # 【矩阵标签】右侧的行标题 (FOV)
            if j == 3:
                ax.text(1.05, 0.5, f"FOV = {fov}°", transform=ax.transAxes, 
                        rotation=-90, va='center', ha='left', 
                        weight='bold', fontsize=13, color='#333333')

    # 全局坐标轴标签 (调整了相对位置以适应宽图)
    fig.supylabel(r"Accuracy $a_{1.25}$ (%)", fontsize=14, weight='bold', x=0.08, color='#333333')
    fig.supxlabel("Camera Pitch Angle (°)", fontsize=14, weight='bold', y=0.02, color='#333333')

    # 底部居中图例
    custom_lines = [
        Line2D([0], [0], color=COLOR_BASE_120, linestyle='--', marker='o', markersize=7, label='Baseline (120m)'),
        Line2D([0], [0], color=COLOR_BASE_80, linestyle='--', marker='o', markersize=7, label='Baseline (80m)'),
        Line2D([0], [0], color=COLOR_OURS_120, linestyle='-', marker='s', markersize=7, label='Ours (120m)'),
        Line2D([0], [0], color=COLOR_OURS_80, linestyle='-', marker='s', markersize=7, label='Ours (80m)')
    ]
    fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=4, frameon=False, fontsize=12, handletextpad=0.5, columnspacing=3.0)

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 横幅 (2x4) 完美置信区间折线图已生成: {save_path}")

if __name__ == "__main__":
    baseline_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    ours_csv = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    output_png = "3_Lineplots_Grid_Horizontal.png"
    
    df = load_data(baseline_csv, ours_csv)
    plot_lineplots_horizontal_grid(df, output_png)