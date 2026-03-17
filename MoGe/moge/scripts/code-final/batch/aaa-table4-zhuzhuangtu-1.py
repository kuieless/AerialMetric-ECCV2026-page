

# # # # # # import pandas as pd
# # # # # # import numpy as np
# # # # # # import matplotlib.pyplot as plt
# # # # # # import os
# # # # # # import warnings
# # # # # # try:
# # # # # #     import statsmodels.api as sm
# # # # # #     from statsmodels.formula.api import ols
# # # # # # except ImportError:
# # # # # #     print("⚠️ 缺少 statsmodels 库。请在终端运行: pip install statsmodels")
# # # # # #     exit()

# # # # # # warnings.filterwarnings("ignore")

# # # # # # # ================= 学术绘图全局设置 =================
# # # # # # plt.rcParams.update({
# # # # # #     'font.family': 'serif',
# # # # # #     'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
# # # # # #     'mathtext.fontset': 'stix', 
# # # # # #     'font.size': 14,
# # # # # #     'axes.labelsize': 14,
# # # # # #     'xtick.labelsize': 14,
# # # # # #     'ytick.labelsize': 14,
# # # # # #     'legend.fontsize': 13,
# # # # # #     'axes.titlesize': 16, 
# # # # # #     'figure.dpi': 300
# # # # # # })

# # # # # # COLORS = {
# # # # # #     'MoGe2':        '#E64B35',   
# # # # # #     'DepthPro':     '#00A087',   
# # # # # #     'UniDepth':     '#3C5488',   
# # # # # #     'ZoeDepth':     '#F39B7F',   
# # # # # #     'MoGe2-Aerial': '#4DBBD5'    
# # # # # # }
# # # # # # # 所有方法
# # # # # # ALL_METHODS = ['MoGe2', 'DepthPro', 'UniDepth', 'ZoeDepth', 'MoGe2-Aerial']
# # # # # # # ANOVA 分析中剔除完全失效的方法，使得图表更干净
# # # # # # ANOVA_METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']
# # # # # # SCENES = ['Campus', 'Factory', 'Farm', 'Grass']

# # # # # # # ================= 数据读取函数 =================
# # # # # # def load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd):
# # # # # #     df_base = pd.read_csv(baseline_csv)
# # # # # #     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    
# # # # # #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
# # # # # #     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
# # # # # #     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
# # # # # #     def load_txt(txt_path, col_name):
# # # # # #         df_txt = pd.read_csv(txt_path, sep='\t')
# # # # # #         df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
# # # # # #         df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
# # # # # #         df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
# # # # # #         df_sub.rename(columns={'depth_metric.delta1': col_name}, inplace=True)
# # # # # #         return df_sub

# # # # # #     df_depthpro = load_txt(txt_dp, 'DepthPro')
# # # # # #     df_unidepth = load_txt(txt_ud, 'UniDepth')
# # # # # #     df_zoedepth = load_txt(txt_zd, 'ZoeDepth')
    
# # # # # #     for df_method in [df_depthpro, df_unidepth, df_zoedepth]:
# # # # # #         df_merged = pd.merge(df_merged, df_method, on=['Scene', 'Filename'], how='inner')
        
# # # # # #     scene_map = {
# # # # # #         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# # # # # #         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
# # # # # #     }
# # # # # #     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
    
# # # # # #     for m in ALL_METHODS:
# # # # # #         if df_merged[m].max() <= 1.05:
# # # # # #             df_merged[m] *= 100
            
# # # # # #     return df_merged

# # # # # # # ================= 方案一：分场景的 ANOVA 方差归因图 =================
# # # # # # def plot_anova_variance_per_scene(df, output_dir="."):
# # # # # #     fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
# # # # # #     var_colors = ['#2E4053', '#D35400', '#2980B9', '#8E44AD', '#BDC3C7']
    
# # # # # #     for idx, scene in enumerate(SCENES):
# # # # # #         ax = axes[idx // 2, idx % 2]
# # # # # #         df_scene = df[df['Scene'] == scene]
# # # # # #         variance_results = {}
        
# # # # # #         for m in ANOVA_METHODS:
# # # # # #             safe_m = m.replace('-', '_')
# # # # # #             df_temp = df_scene.copy()
# # # # # #             df_temp.rename(columns={m: safe_m}, inplace=True)
            
# # # # # #             formula = f"{safe_m} ~ C(Pitch_Group) + C(Height_Group) + C(FOV_Group) + " \
# # # # # #                       f"C(Pitch_Group):C(Height_Group) + C(Pitch_Group):C(FOV_Group) + " \
# # # # # #                       f"C(Height_Group):C(FOV_Group)"
            
# # # # # #             model = ols(formula, data=df_temp).fit()
# # # # # #             anova_table = sm.stats.anova_lm(model, typ=2)
            
# # # # # #             total_sum_sq = anova_table['sum_sq'].sum()
# # # # # #             if total_sum_sq == 0:
# # # # # #                 eta_sq = pd.Series(0, index=anova_table.index)
# # # # # #             else:
# # # # # #                 eta_sq = (anova_table['sum_sq'] / total_sum_sq) * 100
            
# # # # # #             variance_results[m] = {
# # # # # #                 'Pitch': eta_sq.get('C(Pitch_Group)', 0),
# # # # # #                 'Height': eta_sq.get('C(Height_Group)', 0),
# # # # # #                 'FOV': eta_sq.get('C(FOV_Group)', 0),
# # # # # #                 'Pitch \times Height': eta_sq.get('C(Pitch_Group):C(Height_Group)', 0),
# # # # # #                 'Residuals (Other)': eta_sq.get('Residual', 0) + eta_sq.get('C(Pitch_Group):C(FOV_Group)', 0) + eta_sq.get('C(Height_Group):C(FOV_Group)', 0)
# # # # # #             }
            
# # # # # #         df_var = pd.DataFrame(variance_results).T
        
# # # # # #         df_var.plot(kind='bar', stacked=True, ax=ax, color=var_colors, edgecolor='white', width=0.5, legend=False)
        
# # # # # #         ax.set_title(f'Scene: {scene}', fontsize=16, weight='bold')
# # # # # #         ax.set_ylim(0, 100)
# # # # # #         ax.set_xticklabels(ax.get_xticklabels(), rotation=0, weight='bold')
        
# # # # # #         if idx % 2 == 0:
# # # # # #             ax.set_ylabel('Explained Variance (%)')
            
# # # # # #         # 标注大于 10% 的数值
# # # # # #         for c in ax.containers:
# # # # # #             ax.bar_label(c, fmt=lambda v: f"{v:.1f}%" if v > 10 else "", label_type='center', color='white', weight='bold')

# # # # # #     # 全局图例
# # # # # #     handles, labels = ax.get_legend_handles_labels()
# # # # # #     fig.legend(handles, labels, title='Variance Source', loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=False)
    
# # # # # #     plt.suptitle("Variance Contribution per Scene ($\eta^2$)", fontsize=20, y=1.02, weight='bold')
# # # # # #     plt.tight_layout()
# # # # # #     plt.savefig(os.path.join(output_dir, "Analysis_ANOVA_Per_Scene.png"), dpi=300, bbox_inches='tight')
# # # # # #     plt.close()
# # # # # #     print("✅ 方案一生成：Analysis_ANOVA_Per_Scene.png")

# # # # # # # ================= 方案二：分场景的双变量响应曲面 =================
# # # # # # def plot_response_surface_per_scene(df, output_dir="."):
# # # # # #     compare_methods = ['MoGe2', 'MoGe2-Aerial']
    
# # # # # #     # 4行(Scenes) x 2列(Methods)
# # # # # #     fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharex=True, sharey=True)
    
# # # # # #     for i, scene in enumerate(SCENES):
# # # # # #         df_scene = df[df['Scene'] == scene]
        
# # # # # #         for j, m in enumerate(compare_methods):
# # # # # #             ax = axes[i, j]
            
# # # # # #             df_grid = df_scene.groupby(['Pitch_Group', 'Height_Group'])[m].mean().reset_index()
# # # # # #             X_vals = sorted(df_grid['Pitch_Group'].unique())
# # # # # #             Y_vals = sorted(df_grid['Height_Group'].unique())
            
# # # # # #             X, Y = np.meshgrid(X_vals, Y_vals)
# # # # # #             Z = np.zeros_like(X, dtype=float)
            
# # # # # #             for idx, row in df_grid.iterrows():
# # # # # #                 xi = X_vals.index(row['Pitch_Group'])
# # # # # #                 yi = Y_vals.index(row['Height_Group'])
# # # # # #                 Z[yi, xi] = row[m]
                
# # # # # #             levels = np.linspace(0, 100, 11)
# # # # # #             contour = ax.contourf(X, Y, Z, levels=levels, cmap='Spectral', alpha=0.9, extend='both')
# # # # # #             lines = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
# # # # # #             ax.clabel(lines, inline=True, fontsize=10, fmt='%1.0f')
            
# # # # # #             # 设置标题与标签
# # # # # #             if i == 0:
# # # # # #                 ax.set_title(f"Method: {m}", fontsize=18, weight='bold', color=COLORS[m])
# # # # # #             if i == 3:
# # # # # #                 ax.set_xlabel('Pitch (Degrees)', fontsize=14)
# # # # # #             if j == 0:
# # # # # #                 ax.set_ylabel(f'Scene: {scene}\nAltitude (m)', fontsize=14, weight='bold')
                
# # # # # #             ax.set_xticks(X_vals)
# # # # # #             ax.set_yticks(Y_vals)
# # # # # #             if i == 0 and j == 0:
# # # # # #                 ax.invert_xaxis() # 只需反转一次
            
# # # # # #     # 全局 Colorbar
# # # # # #     cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
# # # # # #     cbar = fig.colorbar(contour, cax=cbar_ax)
# # # # # #     cbar.set_label('delta < 1.25 (%)', rotation=270, labelpad=20, fontsize=14)
    
# # # # # #     plt.tight_layout()
# # # # # #     plt.savefig(os.path.join(output_dir, "Analysis_Response_Surface_Per_Scene.png"), dpi=300, bbox_inches='tight')
# # # # # #     plt.close()
# # # # # #     print("✅ 方案二生成：Analysis_Response_Surface_Per_Scene.png")

# # # # # # # ================= 主运行入口 =================
# # # # # # if __name__ == "__main__":
# # # # # #     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# # # # # #     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# # # # # #     txt_dp = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/ml_depth_pro_Bench_per_image.txt"
# # # # # #     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"
# # # # # #     txt_zd = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/zoedepth_nk_metric_benchmarks_Bench_per_image.txt"

# # # # # #     print("⏳ 正在读取数据...")
# # # # # #     df = load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd)
    
# # # # # #     print("⏳ 开始生成分场景深度分析图表...")
# # # # # #     plot_anova_variance_per_scene(df)
# # # # # #     plot_response_surface_per_scene(df)
# # # # # #     print("🎉 分析执行完毕！请查看生成的 2 张矩阵图片。")


















# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # import matplotlib.pyplot as plt
# # # # # import os
# # # # # import warnings
# # # # # try:
# # # # #     import statsmodels.api as sm
# # # # #     from statsmodels.formula.api import ols
# # # # # except ImportError:
# # # # #     print("⚠️ 缺少 statsmodels 库。请在终端运行: pip install statsmodels")
# # # # #     exit()

# # # # # warnings.filterwarnings("ignore")

# # # # # # ================= 学术绘图全局设置 =================
# # # # # plt.rcParams.update({
# # # # #     'font.family': 'serif',
# # # # #     'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
# # # # #     'mathtext.fontset': 'stix', 
# # # # #     'font.size': 14,
# # # # #     'axes.labelsize': 14,
# # # # #     'xtick.labelsize': 14,
# # # # #     'ytick.labelsize': 14,
# # # # #     'legend.fontsize': 12, # 图例变多了，字号稍微调小一点点
# # # # #     'axes.titlesize': 16, 
# # # # #     'figure.dpi': 300
# # # # # })

# # # # # COLORS = {
# # # # #     'MoGe2':        '#E64B35',   
# # # # #     'DepthPro':     '#00A087',   
# # # # #     'UniDepth':     '#3C5488',   
# # # # #     'ZoeDepth':     '#F39B7F',   
# # # # #     'MoGe2-Aerial': '#4DBBD5'    
# # # # # }
# # # # # ALL_METHODS = ['MoGe2', 'DepthPro', 'UniDepth', 'ZoeDepth', 'MoGe2-Aerial']
# # # # # # 同样剔除彻底崩盘的两个方法
# # # # # ANOVA_METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']
# # # # # SCENES = ['Campus', 'Factory', 'Farm', 'Grass']

# # # # # def load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd):
# # # # #     df_base = pd.read_csv(baseline_csv)
# # # # #     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    
# # # # #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
# # # # #     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
# # # # #     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
# # # # #     def load_txt(txt_path, col_name):
# # # # #         df_txt = pd.read_csv(txt_path, sep='\t')
# # # # #         df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
# # # # #         df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
# # # # #         df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
# # # # #         df_sub.rename(columns={'depth_metric.delta1': col_name}, inplace=True)
# # # # #         return df_sub

# # # # #     df_depthpro = load_txt(txt_dp, 'DepthPro')
# # # # #     df_unidepth = load_txt(txt_ud, 'UniDepth')
# # # # #     df_zoedepth = load_txt(txt_zd, 'ZoeDepth')
    
# # # # #     for df_method in [df_depthpro, df_unidepth, df_zoedepth]:
# # # # #         df_merged = pd.merge(df_merged, df_method, on=['Scene', 'Filename'], how='inner')
        
# # # # #     scene_map = {
# # # # #         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# # # # #         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
# # # # #     }
# # # # #     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
    
# # # # #     for m in ALL_METHODS:
# # # # #         if df_merged[m].max() <= 1.05:
# # # # #             df_merged[m] *= 100
            
# # # # #     return df_merged

# # # # # # ================= 全变量拆解的 ANOVA 方差归因图 =================
# # # # # def plot_anova_full_variance_per_scene(df, output_dir="."):
# # # # #     fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=True)
    
# # # # #     # 为 8 个变量分配颜色 (主效应深色，交互效应亮色，残差灰色)
# # # # #     var_colors = [
# # # # #         '#1F618D', # Pitch (深蓝)
# # # # #         '#B03A2E', # Height (深红)
# # # # #         '#1E8449', # FOV (深绿)
# # # # #         '#AF7AC5', # Pitch x Height (浅紫)
# # # # #         '#F1C40F', # Pitch x FOV (亮黄)
# # # # #         '#E67E22', # Height x FOV (亮橙)
# # # # #         '#E74C3C', # Pitch x Height x FOV (亮红)
# # # # #         '#BDC3C7'  # Residuals (高级灰)
# # # # #     ]
    
# # # # #     for idx, scene in enumerate(SCENES):
# # # # #         ax = axes[idx // 2, idx % 2]
# # # # #         df_scene = df[df['Scene'] == scene]
# # # # #         variance_results = {}
        
# # # # #         for m in ANOVA_METHODS:
# # # # #             safe_m = m.replace('-', '_')
# # # # #             df_temp = df_scene.copy()
# # # # #             df_temp.rename(columns={m: safe_m}, inplace=True)
            
# # # # #             # 使用星号 (*) 会让 statsmodels 自动展开所有的主效应、二阶和三阶交互项
# # # # #             formula = f"{safe_m} ~ C(Pitch_Group) * C(Height_Group) * C(FOV_Group)"
            
# # # # #             model = ols(formula, data=df_temp).fit()
# # # # #             anova_table = sm.stats.anova_lm(model, typ=2)
            
# # # # #             total_sum_sq = anova_table['sum_sq'].sum()
# # # # #             if total_sum_sq == 0:
# # # # #                 eta_sq = pd.Series(0, index=anova_table.index)
# # # # #             else:
# # # # #                 eta_sq = (anova_table['sum_sq'] / total_sum_sq) * 100
            
# # # # #             # 精确提取所有的 8 个效应
# # # # #             variance_results[m] = {
# # # # #                 'Pitch': eta_sq.get('C(Pitch_Group)', 0),
# # # # #                 'Height': eta_sq.get('C(Height_Group)', 0),
# # # # #                 'FOV': eta_sq.get('C(FOV_Group)', 0),
# # # # #                 'Pitch x Height': eta_sq.get('C(Pitch_Group):C(Height_Group)', 0),
# # # # #                 'Pitch x FOV': eta_sq.get('C(Pitch_Group):C(FOV_Group)', 0),
# # # # #                 'Height x FOV': eta_sq.get('C(Height_Group):C(FOV_Group)', 0),
# # # # #                 'Pitch x Height x FOV': eta_sq.get('C(Pitch_Group):C(Height_Group):C(FOV_Group)', 0),
# # # # #                 'Pure Residuals': eta_sq.get('Residual', 0)
# # # # #             }
            
# # # # #         df_var = pd.DataFrame(variance_results).T
        
# # # # #         # 绘制
# # # # #         df_var.plot(kind='bar', stacked=True, ax=ax, color=var_colors, edgecolor='white', width=0.5, legend=False)
        
# # # # #         ax.set_title(f'Scene: {scene}', fontsize=16, weight='bold')
# # # # #         ax.set_ylim(0, 100)
# # # # #         ax.set_xticklabels(ax.get_xticklabels(), rotation=0, weight='bold')
        
# # # # #         if idx % 2 == 0:
# # # # #             ax.set_ylabel('Explained Variance (%)')
            
# # # # #         # 标注大于 5% 的数值 (因为切得太细了，阈值放宽到 5%)
# # # # #         for c in ax.containers:
# # # # #             ax.bar_label(c, fmt=lambda v: f"{v:.1f}%" if v > 5 else "", label_type='center', color='white', weight='bold', fontsize=11)

# # # # #     # 全局图例
# # # # #     handles, labels = ax.get_legend_handles_labels()
# # # # #     fig.legend(handles, labels, title='Variance Source (Main Effects & Interactions)', 
# # # # #                loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False, title_fontsize=14)
    
# # # # #     plt.suptitle("Complete Variance Breakdown of Orthogonal Factors ($\eta^2$)", fontsize=20, y=1.02, weight='bold')
# # # # #     plt.tight_layout()
# # # # #     plt.savefig(os.path.join(output_dir, "Analysis_ANOVA_Full_Breakdown.png"), dpi=300, bbox_inches='tight')
# # # # #     plt.close()
# # # # #     print("✅ 全变量拆解图已生成：Analysis_ANOVA_Full_Breakdown.png")

# # # # # if __name__ == "__main__":
# # # # #     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# # # # #     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# # # # #     txt_dp = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/ml_depth_pro_Bench_per_image.txt"
# # # # #     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"
# # # # #     txt_zd = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/zoedepth_nk_metric_benchmarks_Bench_per_image.txt"

# # # # #     print("⏳ 正在读取数据...")
# # # # #     df = load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd)
    
# # # # #     print("⏳ 开始生成全变量 ANOVA 拆解图...")
# # # # #     plot_anova_full_variance_per_scene(df)
# # # # #     print("🎉 图表生成完毕！快去看看这口锅到底是怎么分的吧！")








# # # # import pandas as pd
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # import os
# # # # import warnings
# # # # try:
# # # #     import statsmodels.api as sm
# # # #     from statsmodels.formula.api import ols
# # # # except ImportError:
# # # #     print("⚠️ 缺少 statsmodels 库。请在终端运行: pip install statsmodels")
# # # #     exit()

# # # # warnings.filterwarnings("ignore")

# # # # # ================= 学术绘图全局设置 =================
# # # # plt.rcParams.update({
# # # #     'font.family': 'serif',
# # # #     'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
# # # #     'mathtext.fontset': 'stix', 
# # # #     'font.size': 14,
# # # #     'axes.labelsize': 14,
# # # #     'xtick.labelsize': 14,
# # # #     'ytick.labelsize': 14,
# # # #     'legend.fontsize': 13,
# # # #     'axes.titlesize': 16, 
# # # #     'figure.dpi': 300
# # # # })

# # # # COLORS = {
# # # #     'MoGe2':        '#E64B35',   
# # # #     'DepthPro':     '#00A087',   
# # # #     'UniDepth':     '#3C5488',   
# # # #     'ZoeDepth':     '#F39B7F',   
# # # #     'MoGe2-Aerial': '#4DBBD5'    
# # # # }
# # # # ALL_METHODS = ['MoGe2', 'DepthPro', 'UniDepth', 'ZoeDepth', 'MoGe2-Aerial']
# # # # ANOVA_METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']
# # # # SCENES = ['Campus', 'Factory', 'Farm', 'Grass']

# # # # # ================= 数据读取函数 =================
# # # # def load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd):
# # # #     df_base = pd.read_csv(baseline_csv)
# # # #     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    
# # # #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
# # # #     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
# # # #     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
# # # #     def load_txt(txt_path, col_name):
# # # #         df_txt = pd.read_csv(txt_path, sep='\t')
# # # #         df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
# # # #         df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
# # # #         df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
# # # #         df_sub.rename(columns={'depth_metric.delta1': col_name}, inplace=True)
# # # #         return df_sub

# # # #     df_depthpro = load_txt(txt_dp, 'DepthPro')
# # # #     df_unidepth = load_txt(txt_ud, 'UniDepth')
# # # #     df_zoedepth = load_txt(txt_zd, 'ZoeDepth')
    
# # # #     for df_method in [df_depthpro, df_unidepth, df_zoedepth]:
# # # #         df_merged = pd.merge(df_merged, df_method, on=['Scene', 'Filename'], how='inner')
        
# # # #     scene_map = {
# # # #         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# # # #         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
# # # #     }
# # # #     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
    
# # # #     for m in ALL_METHODS:
# # # #         if df_merged[m].max() <= 1.05:
# # # #             df_merged[m] *= 100
            
# # # #     return df_merged

# # # # # ================= 方案二：分场景的双变量响应曲面 (更新为 3 列) =================
# # # # def plot_response_surface_per_scene(df, output_dir="."):
# # # #     # 核心修改点：加入 UniDepth
# # # #     compare_methods = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']
    
# # # #     # 修改布局为 4行 x 3列，宽度从 14 增加到 21 保持比例
# # # #     fig, axes = plt.subplots(4, 3, figsize=(21, 20), sharex=True, sharey=True)
    
# # # #     for i, scene in enumerate(SCENES):
# # # #         df_scene = df[df['Scene'] == scene]
        
# # # #         for j, m in enumerate(compare_methods):
# # # #             ax = axes[i, j]
            
# # # #             # 数据聚合与网格化
# # # #             df_grid = df_scene.groupby(['Pitch_Group', 'Height_Group'])[m].mean().reset_index()
# # # #             X_vals = sorted(df_grid['Pitch_Group'].unique())
# # # #             Y_vals = sorted(df_grid['Height_Group'].unique())
            
# # # #             X, Y = np.meshgrid(X_vals, Y_vals)
# # # #             Z = np.zeros_like(X, dtype=float)
            
# # # #             for idx, row in df_grid.iterrows():
# # # #                 xi = X_vals.index(row['Pitch_Group'])
# # # #                 yi = Y_vals.index(row['Height_Group'])
# # # #                 Z[yi, xi] = row[m]
                
# # # #             # 绘制等高线
# # # #             levels = np.linspace(0, 100, 11)
# # # #             contour = ax.contourf(X, Y, Z, levels=levels, cmap='Spectral', alpha=0.9, extend='both')
# # # #             lines = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
# # # #             ax.clabel(lines, inline=True, fontsize=10, fmt='%1.0f')
            
# # # #             # 设置标题 (第一行显示方法名)
# # # #             if i == 0:
# # # #                 ax.set_title(f"Method: {m}", fontsize=20, weight='bold', color=COLORS.get(m, 'black'))
            
# # # #             # 设置底部 X 轴标签
# # # #             if i == 3:
# # # #                 ax.set_xlabel('Pitch (Degrees)', fontsize=16)
            
# # # #             # 设置左侧 Y 轴场景标签
# # # #             if j == 0:
# # # #                 ax.set_ylabel(f'Scene: {scene}\nAltitude (m)', fontsize=16, weight='bold')
                
# # # #             ax.set_xticks(X_vals)
# # # #             ax.set_yticks(Y_vals)
            
# # # #             # 只对第一个子图执行一次反转，sharex 会同步所有子图
# # # #             if i == 0 and j == 0:
# # # #                 ax.invert_xaxis() 
            
# # # #     # 全局 Colorbar 位置微调
# # # #     cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # 调整 left 位置防止遮挡第三列
# # # #     cbar = fig.colorbar(contour, cax=cbar_ax)
# # # #     cbar.set_label('delta < 1.25 (%)', rotation=270, labelpad=20, fontsize=16)
    
# # # #     plt.tight_layout(rect=[0, 0, 0.9, 1]) # 为右侧 colorbar 预留空间
# # # #     save_path = os.path.join(output_dir, "Analysis_Response_Surface_Per_Scene_3Methods.png")
# # # #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
# # # #     plt.close()
# # # #     print(f"✅ 方案二生成：{save_path}")

# # # # # ================= 主运行入口 =================
# # # # if __name__ == "__main__":
# # # #     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# # # #     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# # # #     txt_dp = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/ml_depth_pro_Bench_per_image.txt"
# # # #     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"
# # # #     txt_zd = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/zoedepth_nk_metric_benchmarks_Bench_per_image.txt"

# # # #     print("⏳ 正在读取数据...")
# # # #     df = load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd)
    
# # # #     print("⏳ 开始生成包含 UniDepth 的分场景深度分析图表...")
# # # #     plot_response_surface_per_scene(df)
# # # #     print("🎉 分析执行完毕！")


# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import os
# # # import warnings

# # # warnings.filterwarnings("ignore")

# # # # ================= 学术绘图全局设置 =================
# # # plt.rcParams.update({
# # #     'font.family': 'serif',
# # #     'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
# # #     'mathtext.fontset': 'stix', 
# # #     'font.size': 12,
# # #     'axes.labelsize': 14,
# # #     'xtick.labelsize': 12,
# # #     'ytick.labelsize': 12,
# # #     'legend.fontsize': 12,
# # #     'figure.dpi': 300
# # # })

# # # COLORS_DICT = {'MoGe2': '#E64B35', 'UniDepth': '#3C5488', 'MoGe2-Aerial': '#4DBBD5'}
# # # METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']

# # # # ================= 数据读取函数 (保持一致) =================
# # # def load_all_data(baseline_csv, ours_csv, txt_dp, txt_ud, txt_zd):
# # #     df_base = pd.read_csv(baseline_csv)
# # #     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
# # #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
# # #     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
# # #     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
# # #     def load_txt(txt_path, col_name):
# # #         df_txt = pd.read_csv(txt_path, sep='\t')
# # #         df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
# # #         df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
# # #         df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
# # #         df_sub.rename(columns={'depth_metric.delta1': col_name}, inplace=True)
# # #         return df_sub

# # #     df_unidepth = load_txt(txt_ud, 'UniDepth')
# # #     df_merged = pd.merge(df_merged, df_unidepth, on=['Scene', 'Filename'], how='inner')
    
# # #     scene_map = {
# # #         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# # #         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
# # #     }
# # #     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
# # #     for m in METHODS:
# # #         if df_merged[m].max() <= 1.05: df_merged[m] *= 100
# # #     return df_merged

# # # # ================= 1. FOV 鲁棒性对比柱状图 =================
# # # def plot_fov_bar_chart(df, output_path):
# # #     # 计算全场景平均
# # #     df_fov = df.groupby('FOV_Group')[METHODS].mean().reset_index()
    
# # #     # 转换为长格式以便绘图
# # #     df_plot = df_fov.melt(id_vars='FOV_Group', var_name='Method', value_name='delta1')
    
# # #     fig, ax = plt.subplots(figsize=(8, 6))
# # #     sns.barplot(data=df_plot, x='FOV_Group', y='delta1', hue='Method', 
# # #                 palette=list(COLORS_DICT.values()), edgecolor='black', ax=ax)
    
# # #     ax.set_title('Global Performance Comparison across FOV Groups', fontsize=16, weight='bold', pad=15)
# # #     ax.set_xlabel('FOV (Degrees)', fontsize=14)
# # #     ax.set_ylabel(r'$\delta < 1.25$ (%)', fontsize=14)
# # #     ax.set_ylim(0, 105)
# # #     ax.grid(axis='y', linestyle='--', alpha=0.7)
    
# # #     plt.savefig(output_path, bbox_inches='tight')
# # #     plt.close()
# # #     print(f"✅ FOV 柱状图已保存: {output_path}")

# # # # ================= 2. 飞行包络性能矩阵 (Heatmap) =================
# # # def plot_performance_matrices(df, scene_name="Global"):
# # #     if scene_name != "Global":
# # #         data = df[df['Scene'] == scene_name]
# # #     else:
# # #         data = df

# # #     fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    
# # #     # 统一色阶 0-100，使用 RdYlGn (红-黄-绿)
# # #     cmap = "RdYlGn"
    
# # #     for i, m in enumerate(METHODS):
# # #         # 创建透视表：横轴 Pitch，纵轴 Height
# # #         matrix = data.groupby(['Height_Group', 'Pitch_Group'])[m].mean().unstack()
# # #         # 排序确保顺序正确
# # #         matrix = matrix.sort_index(ascending=False) # 高度从高到低 (120, 80)
# # #         matrix = matrix.reindex(columns=[-90, -75, -60, -45]) # 角度从垂直向倾斜

# # #         sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, vmin=0, vmax=100, 
# # #                     ax=axes[i], cbar=(i == 2), annot_kws={"size": 12, "weight": "bold"})
        
# # #         axes[i].set_title(f"Method: {m}", fontsize=16, weight='bold')
# # #         axes[i].set_xlabel('Pitch Angle (Degrees)', fontsize=12)
# # #         if i == 0:
# # #             axes[i].set_ylabel('Altitude (m)', fontsize=12)
# # #         else:
# # #             axes[i].set_ylabel('')

# # #     plt.suptitle(f'Flight Envelope Performance Matrix: {scene_name}', fontsize=20, y=1.05, weight='bold')
# # #     plt.tight_layout()
# # #     filename = f"Performance_Matrix_{scene_name}.png"
# # #     plt.savefig(filename, bbox_inches='tight')
# # #     plt.close()
# # #     print(f"✅ 性能矩阵 [{scene_name}] 已保存: {filename}")

# # # # ================= 主程序 =================
# # # if __name__ == "__main__":
# # #     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# # #     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# # #     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

# # #     df = load_all_data(baseline_csv, ours_csv, None, txt_ud, None)

# # #     # 1. 绘制全局 FOV 柱状图
# # #     plot_fov_bar_chart(df, "Global_FOV_Comparison_Bar.png")

# # #     # 2. 绘制性能矩阵
# # #     # 全局总表
# # #     plot_performance_matrices(df, "Global")
# # #     # 分场景表
# # #     for scene in ['Campus', 'Factory', 'Farm', 'Grass']:
# # #         plot_performance_matrices(df, scene)

# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import os
# # import warnings

# # warnings.filterwarnings("ignore")

# # # ================= 学术绘图全局设置 (适配 12 号以上字号) =================
# # plt.rcParams.update({
# #     'font.family': 'serif',
# #     'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
# #     'mathtext.fontset': 'stix', 
# #     'font.size': 14,
# #     'axes.labelsize': 18,
# #     'xtick.labelsize': 16,
# #     'ytick.labelsize': 16,
# #     'legend.fontsize': 16,
# #     'figure.dpi': 300
# # })

# # METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']

# # # ================= 数据读取函数 =================
# # def load_all_data(baseline_csv, ours_csv, txt_ud):
# #     df_base = pd.read_csv(baseline_csv)
# #     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
# #     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
# #     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
# #     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
# #     # 读取 UniDepth
# #     df_txt = pd.read_csv(txt_ud, sep='\t')
# #     df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
# #     df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
# #     df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
# #     df_sub.rename(columns={'depth_metric.delta1': 'UniDepth'}, inplace=True)
# #     df_merged = pd.merge(df_merged, df_sub, on=['Scene', 'Filename'], how='inner')
    
# #     scene_map = {
# #         'Cleaned_Dataset_Campus': 'Campus', 'Cleaned_Dataset_Factory': 'Factory',
# #         'Cleaned_Dataset_Farm': 'Farm', 'Cleaned_Dataset_Grass': 'Grass', 'Cleaned_Dataset_Gress': 'Grass'
# #     }
# #     df_merged['Scene'] = df_merged['Scene'].map(scene_map)
# #     for m in METHODS:
# #         if df_merged[m].max() <= 1.05: df_merged[m] *= 100
# #     return df_merged

# # # ================= 1. FOV 柱状图 (规整版) =================
# # def plot_fov_bar_chart(df, output_path):
# #     df_fov_global = df.groupby('FOV_Group')[METHODS].mean().reset_index()
# #     df_plot = df_fov_global.melt(id_vars='FOV_Group', var_name='Method', value_name='delta1')
    
# #     fig, ax = plt.subplots(figsize=(12, 7))
# #     # 横轴改为 Method，图例改为 FOV
# #     sns.barplot(data=df_plot, x='Method', y='delta1', hue='FOV_Group', 
# #                 palette=['#7FB3D5', '#2980B9'], edgecolor='black', ax=ax)
    
# #     ax.set_title('Metric Accuracy ($\delta < 1.25$) across FOV Groups', fontsize=22, weight='bold', pad=25)
# #     ax.set_xlabel('Method', fontsize=20, weight='bold')
# #     ax.set_ylabel('Accuracy (%)', fontsize=20, weight='bold')
# #     ax.set_ylim(0, 115)
# #     ax.legend(title='FOV (Degrees)', loc='upper left', frameon=True)
# #     ax.grid(axis='y', linestyle='--', alpha=0.5)
    
# #     # 数值标注
# #     for p in ax.patches:
# #         ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
# #                     ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', 
# #                     fontsize=14, weight='bold')

# #     plt.savefig(output_path, bbox_inches='tight')
# #     plt.close()

# # # ================= 2. 性能矩阵 Heatmap (方正规整版) =================
# # def plot_performance_matrices(df, scene_name="Global"):
# #     data_sub = df if scene_name == "Global" else df[df['Scene'] == scene_name]

# #     fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
# #     cmap = "RdYlGn"
    
# #     for i, m in enumerate(METHODS):
# #         matrix = data_sub.groupby(['Height_Group', 'Pitch_Group'])[m].mean().unstack()
# #         matrix = matrix.sort_index(ascending=False) # 高度 120 在上
# #         matrix = matrix.reindex(columns=[-90, -75, -60, -45])

# #         # square=True 让格子方方正正, annot_kws 调大内部数字
# #         sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, vmin=0, vmax=100, 
# #                     ax=axes[i], cbar=(i == 2), square=True, linewidths=1.5,
# #                     annot_kws={"size": 18, "weight": "bold"},
# #                     cbar_kws={'label': r'Accuracy ($\delta < 1.25$ %)'} if i==2 else None)
        
# #         axes[i].set_title(f"[{m}]", fontsize=22, weight='bold', pad=15)
# #         axes[i].set_xlabel('Pitch Angle (Deg)', fontsize=18)
# #         if i == 0:
# #             axes[i].set_ylabel('Altitude (m)', fontsize=18)
        
# #         axes[i].tick_params(axis='both', which='major', labelsize=16)

# #     plt.suptitle(f'Flight Envelope Performance Matrix ({scene_name})', fontsize=28, y=1.02, weight='bold')
# #     plt.tight_layout(rect=[0, 0, 0.95, 1])
# #     filename = f"Heatmap_Regular_{scene_name}.png"
# #     plt.savefig(filename, bbox_inches='tight')
# #     plt.close()
# #     print(f"✅ 已生成规整图表: {filename}")

# # # ================= 主程序执行 =================
# # if __name__ == "__main__":
# #     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
# #     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
# #     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

# #     df = load_all_data(baseline_csv, ours_csv, txt_ud)

# #     # 1. 规整版柱状图
# #     plot_fov_bar_chart(df, "Global_FOV_Grouped_By_Method.png")

# #     # 2. 规整版性能矩阵 (总+分场景)
# #     plot_performance_matrices(df, "Global")
# #     for scene in ['Campus', 'Factory', 'Farm', 'Grass']:
# #         plot_performance_matrices(df, scene)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import warnings

# warnings.filterwarnings("ignore")

# # ================= 学术绘图全局设置 =================
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
#     'mathtext.fontset': 'stix', 
#     'font.size': 14,
#     'axes.labelsize': 18,
#     'xtick.labelsize': 16,
#     'ytick.labelsize': 16,
#     'legend.fontsize': 16,
#     'figure.dpi': 300
# })

# METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']

# # ================= 数据读取与对齐函数 =================
# def load_all_data(baseline_csv, ours_csv, txt_ud):
#     # 1. 读取 CSV 基础数据
#     df_base = pd.read_csv(baseline_csv)
#     keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
#     df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
#     df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
#     df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
#     # 2. 读取 UniDepth (TXT 格式)
#     df_txt = pd.read_csv(txt_ud, sep='\t')
#     df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
#     df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
#     df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
#     df_sub.rename(columns={'depth_metric.delta1': 'UniDepth'}, inplace=True)
    
#     # 3. 合并所有方法
#     df_merged = pd.merge(df_merged, df_sub, on=['Scene', 'Filename'], how='inner')
    
#     # 统一单位为百分比
#     for m in METHODS:
#         if df_merged[m].max() <= 1.05: 
#             df_merged[m] *= 100
            
#     return df_merged

# # ================= 核心绘图函数：FOV 鲁棒性对比柱状图 =================
# def plot_fov_bar_chart(df, output_path):
#     # 计算全场景、全高度、全角度的平均值 (只按 FOV 分组)
#     df_fov_global = df.groupby('FOV_Group')[METHODS].mean().reset_index()
    
#     # 将数据转换为长格式 (Long-format) 适配 seaborn
#     df_plot = df_fov_global.melt(id_vars='FOV_Group', var_name='Method', value_name='delta1')
    
#     # 创建画布
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     # 绘图：x 轴为方法，hue (颜色) 为 FOV
#     # 使用深浅蓝色系：浅蓝代表窄 FOV，深蓝代表宽 FOV
#     sns.barplot(data=df_plot, x='Method', y='delta1', hue='FOV_Group', 
#                 palette=['#7FB3D5', '#2980B9'], edgecolor='black', ax=ax)
    
#     # 图表细节修饰
#     ax.set_title(r'Metric Accuracy ($\delta < 1.25$) Comparison: FOV Sensitivity', 
#                  fontsize=22, weight='bold', pad=25)
#     ax.set_xlabel('Method', fontsize=20, weight='bold')
#     ax.set_ylabel('Accuracy (%)', fontsize=20, weight='bold')
#     ax.set_ylim(0, 115) # 留出顶部空间放标注
    
#     # 图例设置
#     ax.legend(title='FOV (Degrees)', loc='upper left', frameon=True, shadow=False)
    
#     # 背景网格
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
    
#     # 在柱子上方自动标注数值
#     for p in ax.patches:
#         height = p.get_height()
#         if height > 0:
#             ax.annotate(f'{height:.1f}%', 
#                         (p.get_x() + p.get_width() / 2., height),
#                         ha = 'center', va = 'center', 
#                         xytext = (0, 10), 
#                         textcoords = 'offset points', 
#                         fontsize=14, weight='bold')

#     # 保存图片
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✅ 规整版 FOV 柱状图已生成: {output_path}")

# # ================= 主运行入口 =================
# if __name__ == "__main__":
#     # 文件路径
#     baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
#     ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
#     txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

#     # 加载数据
#     df = load_all_data(baseline_csv, ours_csv, txt_ud)

#     # 绘图
#     plot_fov_bar_chart(df, "Global_FOV_Method_Comparison_Final.png")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# =====================================================================
# 🎨 极致学术配置中心 (Science/Nature/Origin 风格)
# =====================================================================
CONFIG = {
    # --- 字体与字号 (针对双栏论文缩放进行了极致放大) ---
    'font.family': 'serif',
    'axes.labelsize': 32,             # Y轴标签 (Accuracy %)
    'xtick.labelsize': 28,            # X轴方法名称
    'ytick.labelsize': 26,            # Y轴刻度数值
    'legend.fontsize': 24,            # 图例
    'value.fontsize': 20,             # 柱子顶部标注数值
    
    # --- 柱子与配色 (专业色调) ---
    'bar.width': 0.16,                
    'bar.spacing': 0.03,              
    'colors': ['#85C1E9', '#1A5276', "#F2DEAA", "#E9B021"], 
    'edgecolor': 'black',             
    'edgewidth': 2.0,                 
    
    # --- 坐标轴与刻度 (Origin 经典强轮廓) ---
    'axes.linewidth': 3.0,            # 坐标轴外框粗细
    'tick.major.width': 3.0,          # 主刻度线粗细
    'tick.major.size': 12,            # 主刻度线长度
    'tick.direction': 'in',           # 刻度线朝内
    
    # --- 保存设置 ---
    'save_dpi': 600,                  # PNG 分辨率
}

# ================= 学术绘图全局初始化 =================
plt.rcParams.update({
    'font.family': CONFIG['font.family'],
    'font.serif': ['STIXGeneral', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'stix', 
    'pdf.fonttype': 42,               # 确保 PDF 字体嵌入
    'ps.fonttype': 42,
    'figure.dpi': 300
})

METHODS = ['MoGe2', 'UniDepth', 'MoGe2-Aerial']
def load_all_data(baseline_csv, ours_csv, txt_ud):
    # 1. 读取 CSV 基础数据
    df_base = pd.read_csv(baseline_csv)
    keys = ['Scene', 'Filename', 'FOV_Group', 'Height_Group', 'Pitch_Group']
    df_b = df_base[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2'})
    df_o = pd.read_csv(ours_csv)[keys + ['a1.25']].rename(columns={'a1.25': 'MoGe2-Aerial'})
    df_merged = pd.merge(df_b, df_o, on=keys, how='inner')
    
    # 2. 读取 UniDepth (TXT 格式)
    df_txt = pd.read_csv(txt_ud, sep='\t')
    df_txt['Scene'] = df_txt['filename'].apply(lambda x: str(x).split('/')[0])
    df_txt['Filename'] = df_txt['filename'].apply(lambda x: str(x).split('/')[-1])
    df_sub = df_txt[['Scene', 'Filename', 'depth_metric.delta1']].copy()
    df_sub.rename(columns={'depth_metric.delta1': 'UniDepth'}, inplace=True)
    
    # 3. 合并所有方法
    df_merged = pd.merge(df_merged, df_sub, on=['Scene', 'Filename'], how='inner')
    
    # 统一单位为百分比
    for m in METHODS:
        if df_merged[m].max() <= 1.05: 
            df_merged[m] *= 100
            
    return df_merged
# ================= 数据加工：将 Height 改为 Altitude =================
def extract_plot_data(df):
    """提取 FOV 和 Altitude (原 Height) 数据"""
    results = []
    # 1. FOV 数据
    df_fov = df.groupby('FOV_Group')[METHODS].mean().reset_index()
    for _, row in df_fov.iterrows():
        results.append({'Category': f"FOV {row['FOV_Group']}°", **{m: row[m] for m in METHODS}})
        
    # 2. Altitude 数据 (核心更名点)
    if 'Height_Group' in df.columns:
        df_h = df.groupby('Height_Group')[METHODS].mean().reset_index()
        for _, row in df_h.iterrows():
            if row['Height_Group'] in [80, 120]:
                results.append({'Category': f"Altitude {row['Height_Group']}m", **{m: row[m] for m in METHODS}})
                
    return pd.DataFrame(results)

# ================= 核心绘图函数：同时导出 PNG 和 PDF =================
def save_dual_format(fig, base_name):
    """一键保存高保真 PNG 和矢量 PDF"""
    fig.savefig(f"{base_name}.png", dpi=CONFIG['save_dpi'], bbox_inches='tight')
    fig.savefig(f"{base_name}.pdf", format='pdf', bbox_inches='tight')
    print(f"✅ 已成功保存: {base_name}.png & {base_name}.pdf")

def plot_academic_bar_chart(df, base_filename="Method_Comparison_Academic"):
    df_plot = extract_plot_data(df)
    categories = df_plot['Category'].tolist()
    n_categories = len(categories)
    colors = CONFIG['colors']
    x = np.arange(len(METHODS))
    
    # 创建画布 (14x8 是学术插图的黄金比例)
    fig, ax = plt.subplots(figsize=(14, 8)) 
    
    total_width = CONFIG['bar.width'] * n_categories + CONFIG['bar.spacing'] * (n_categories - 1)
    start_offset = -total_width / 2 + CONFIG['bar.width'] / 2
    
    for i, cat in enumerate(categories):
        offset = start_offset + i * (CONFIG['bar.width'] + CONFIG['bar.spacing'])
        values = df_plot[METHODS].iloc[i].values
        bars = ax.bar(x + offset, values, width=CONFIG['bar.width'], label=cat, 
                      color=colors[i], edgecolor=CONFIG['edgecolor'], 
                      linewidth=CONFIG['edgewidth'], zorder=3)
        
        # 柱顶数值标注 (加粗、字号放大)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 10), textcoords="offset points", ha='center', va='bottom',
                            fontsize=CONFIG['value.fontsize'], fontweight='bold')

    # --- 样式极致美化 ---
    # 1. 移除多余的标题和底部标签
    ax.set_title('')
    ax.set_xlabel('')
    
    # 2. 设置 Y 轴标签与字号
    ax.set_ylabel(r'Metric Accuracy ($\delta < 1.25$) %', fontsize=CONFIG['axes.labelsize'], fontweight='bold', labelpad=20)
    
    # 3. 设置刻度与字体
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=CONFIG['xtick.labelsize'], fontweight='bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(CONFIG['ytick.labelsize'])
        label.set_fontweight('bold')
    
    # 4. 🎯 优化：压缩顶部空间 (根据数据调整，115 左右通常最紧凑)
    ax.set_ylim(0, 115) 
    
    # 5. 🎯 优化：图例下移且带边框 (bbox_to_anchor 的 0.88 控制上下)
    ax.legend(
        fontsize=CONFIG['legend.fontsize'], 
        loc='upper left', 
        bbox_to_anchor=(0.02, 0.88), 
        ncol=1, 
        frameon=True, 
        edgecolor='black', 
        fancybox=False, 
        shadow=False
    )
    
    # 6. Origin 风格强边框
    for spine in ax.spines.values():
        spine.set_linewidth(CONFIG['axes.linewidth'])
    ax.tick_params(which='major', width=CONFIG['tick.major.width'], length=CONFIG['tick.major.size'], 
                   direction=CONFIG['tick.direction'], top=True, right=True)
    
    plt.tight_layout()
    save_dual_format(fig, base_filename)
    plt.close()

# ================= 飞行包络矩阵 (同步更新 Altitude) =================
def plot_performance_matrices_pdf(df, scene_name="Global"):
    data_sub = df if scene_name == "Global" else df[df['Scene'] == scene_name]
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
    cmap = "RdYlGn"
    
    for i, m in enumerate(METHODS):
        matrix = data_sub.groupby(['Height_Group', 'Pitch_Group'])[m].mean().unstack()
        matrix = matrix.sort_index(ascending=False)
        matrix = matrix.reindex(columns=[-90, -75, -60, -45])

        sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, vmin=0, vmax=100, 
                    ax=axes[i], cbar=(i == 2), square=True, linewidths=1.5,
                    annot_kws={"size": 18, "weight": "bold"},
                    cbar_kws={'label': r'Accuracy ($\delta < 1.25$ %)'} if i==2 else None)
        
        axes[i].set_title(f"[{m}]", fontsize=22, weight='bold', pad=15)
        axes[i].set_xlabel('Pitch Angle (Deg)', fontsize=18)
        if i == 0:
            axes[i].set_ylabel('Altitude (m)', fontsize=18) # 标签改为 Altitude
        
    plt.suptitle(f'Flight Envelope Matrix ({scene_name})', fontsize=28, y=1.02, weight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    save_dual_format(fig, f"Heatmap_Altitude_{scene_name}")
    plt.close()

# ================= 主运行入口 =================
# if __name__ == "__main__":
#     # ⚠️ 请确保 load_all_data 函数已按照之前的逻辑定义
#     # df = load_all_data(...)
    
#     # 1. 生成精美的 Altitude 鲁棒性柱状图
#     plot_academic_bar_chart(df, "Academic_Altitude_FOV_Comparison")
    
#     # 2. 生成规整的 Altitude 飞行包络矩阵
#     plot_performance_matrices_pdf(df, "Global")

# # ================= 主运行入口 =================
if __name__ == "__main__":
    # 文件路径
    baseline_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    ours_csv = "/data1/szq/inferinfer/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV.csv" 
    txt_ud = "/home/szq/moge2/MoGe/moge/scripts/code-final/batch/images/Fig-bench/unidepth_v2_vitl14_Bench_per_image.txt"

    # 加载数据
    df = load_all_data(baseline_csv, ours_csv, txt_ud)

    # 绘图
    plot_fov_bar_chart(df, "Global_FOV_Method_Comparison_Final.png")