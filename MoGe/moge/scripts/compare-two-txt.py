# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import re
# # # import os

# # # # 设置绘图风格，支持中文显示（如果系统没有中文字体，可能需要调整 font.sans-serif）
# # # sns.set(style="whitegrid")
# # # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# # # def parse_section_data(content, section_title, headers):
# # #     """
# # #     解析指定章节的数据表格
# # #     """
# # #     # 找到章节的开始位置
# # #     start_idx = content.find(section_title)
# # #     if start_idx == -1:
# # #         print(f"Warning: Section '{section_title}' not found.")
# # #         return pd.DataFrame(columns=headers)
    
# # #     # 截取该章节后的内容
# # #     section_content = content[start_idx:]
# # #     lines = section_content.split('\n')
    
# # #     data = []
# # #     parsing = False
# # #     separator_count = 0
    
# # #     for line in lines:
# # #         stripped_line = line.strip()
        
# # #         # 遇到下一个章节（>>>开头）则停止
# # #         if stripped_line.startswith('>>>') and section_title not in stripped_line:
# # #             break
            
# # #         # 跳过空行
# # #         if not stripped_line:
# # #             continue
            
# # #         # 检测分隔线，确定数据开始
# # #         if '----' in stripped_line:
# # #             separator_count += 1
# # #             if separator_count == 2:
# # #                 parsing = True
# # #             continue
            
# # #         if parsing:
# # #             # 使用 | 分割数据
# # #             parts = [p.strip() for p in stripped_line.split('|')]
# # #             # 过滤掉空字符串（行首尾可能产生的空串）
# # #             parts = [p for p in parts if p]
            
# # #             # 简单校验列数，允许最后一列注释可能有差异
# # #             if len(parts) >= len(headers) - 1: # 宽松匹配
# # #                 # 截取需要的列数
# # #                 row_data = parts[:len(headers)]
# # #                 # 如果缺列（比如最后一列为空），补齐
# # #                 if len(row_data) < len(headers):
# # #                     row_data.append("")
# # #                 data.append(row_data)

# # #     df = pd.DataFrame(data, columns=headers)
    
# # #     # 转换数值列
# # #     for col in df.columns:
# # #         if col != 'Scene Name' and col != 'Cat' and col != 'Interp':
# # #             df[col] = pd.to_numeric(df[col], errors='coerce')
            
# # #     return df

# # # def parse_report_file(filepath):
# # #     """
# # #     读取并解析单个报告文件
# # #     """
# # #     if not os.path.exists(filepath):
# # #         print(f"Error: File '{filepath}' not found.")
# # #         return None, None

# # #     with open(filepath, 'r', encoding='utf-8') as f:
# # #         content = f.read()

# # #     # 定义表头
# # #     fov_headers = ['Scene Name', 'Pred FOV', 'GT FOV', 'FOV Err', 'Pred Focal', 'GT Focal', 'Cat']
# # #     scale_headers = ['Scene Name', 'Ratio', 'Std', 'Interp']

# # #     # 解析 FOV 和 Scale 部分
# # #     # 注意：根据你的文件内容，关键词可能是中文或英文，这里做模糊匹配
# # #     df_fov = parse_section_data(content, "Intrinsics Detail Report", fov_headers)
# # #     df_scale = parse_section_data(content, "Scale Drift Analysis", scale_headers)
    
# # #     return df_fov, df_scale

# # # def compare_and_plot(file1, file2):
# # #     """
# # #     对比两份报告并绘图
# # #     """
# # #     print(f"Parsing {file1}...")
# # #     fov1, scale1 = parse_report_file(file1)
# # #     print(f"Parsing {file2}...")
# # #     fov2, scale2 = parse_report_file(file2)

# # #     if fov1 is None or fov2 is None:
# # #         return

# # #     # --- 1. FOV 对比 (FOV Error) ---
# # #     # 合并数据
# # #     fov_merged = pd.merge(fov1[['Scene Name', 'FOV Err', 'Cat']], 
# # #                           fov2[['Scene Name', 'FOV Err']], 
# # #                           on='Scene Name', suffixes=('_1', '_2'))
    
# # #     # 计算差异
# # #     fov_merged['Diff'] = fov_merged['FOV Err_2'] - fov_merged['FOV Err_1']
    
# # #     # 绘图 1: FOV Error 对比散点图
# # #     plt.figure(figsize=(10, 6))
# # #     sns.scatterplot(data=fov_merged, x='FOV Err_1', y='FOV Err_2', hue='Cat', s=100)
    
# # #     # 画对角线（无变化线）
# # #     max_val = max(fov_merged['FOV Err_1'].max(), fov_merged['FOV Err_2'].max())
# # #     plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No Change')
    
# # #     plt.title(f'FOV Error Comparison\n({file1} vs {file2})')
# # #     plt.xlabel(f'FOV Error in {file1}')
# # #     plt.ylabel(f'FOV Error in {file2}')
# # #     plt.legend()
# # #     plt.tight_layout()
# # #     plt.savefig('comparison_fov_scatter.png')
# # #     print("Generated: comparison_fov_scatter.png")

# # #     # 绘图 2: FOV Error 差异最大的前20个场景 (柱状图)
# # #     # 按差异绝对值排序
# # #     fov_merged['AbsDiff'] = fov_merged['Diff'].abs()
# # #     top_diff_fov = fov_merged.sort_values('AbsDiff', ascending=False).head(20)
    
# # #     plt.figure(figsize=(12, 6))
# # #     # sns.barplot(data=top_diff_fov, x='Scene Name', y='Diff', palette='coolwarm')
# # #     # 新的写法（消除警告）
# # #     sns.barplot(data=top_diff_fov, x='Scene Name', y='Diff', hue='Scene Name', palette='coolwarm', legend=False)
# # #     plt.axhline(0, color='black', linewidth=0.8)
# # #     plt.xticks(rotation=45, ha='right')
# # #     plt.title('Top 20 FOV Error Changes (Report 2 - Report 1)\nPositive means Error Increased')
# # #     plt.tight_layout()
# # #     plt.savefig('comparison_fov_diff_bar.png')
# # #     print("Generated: comparison_fov_diff_bar.png")

# # #     # --- 2. Scale 对比 (Ratio) ---
# # #     if not scale1.empty and not scale2.empty:
# # #         scale_merged = pd.merge(scale1[['Scene Name', 'Ratio']], 
# # #                                 scale2[['Scene Name', 'Ratio']], 
# # #                                 on='Scene Name', suffixes=('_1', '_2'))
        
# # #         scale_merged['Diff'] = scale_merged['Ratio_2'] - scale_merged['Ratio_1']

# # #         # 绘图 3: Scale Ratio 对比散点图
# # #         plt.figure(figsize=(10, 6))
# # #         sns.scatterplot(data=scale_merged, x='Ratio_1', y='Ratio_2', s=100, color='purple')
        
# # #         # 画对角线
# # #         min_val = min(scale_merged['Ratio_1'].min(), scale_merged['Ratio_2'].min())
# # #         max_val = max(scale_merged['Ratio_1'].max(), scale_merged['Ratio_2'].max())
# # #         plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No Change')
        
# # #         plt.title(f'Scale Ratio Comparison\n({file1} vs {file2})')
# # #         plt.xlabel(f'Ratio in {file1}')
# # #         plt.ylabel(f'Ratio in {file2}')
# # #         plt.legend()
# # #         plt.tight_layout()
# # #         plt.savefig('comparison_scale_scatter.png')
# # #         print("Generated: comparison_scale_scatter.png")
        
# # #         # 绘图 4: Scale Ratio 差异分布
# # #         plt.figure(figsize=(12, 6))
# # #         # 排序便于观察
# # #         scale_merged_sorted = scale_merged.sort_values('Diff')
# # #         sns.barplot(data=scale_merged_sorted, x='Scene Name', y='Diff', color='skyblue')
# # #         plt.xticks(rotation=90, fontsize=8)
# # #         plt.title('Scale Ratio Changes per Scene (Report 2 - Report 1)')
# # #         plt.tight_layout()
# # #         plt.savefig('comparison_scale_diff_all.png')
# # #         print("Generated: comparison_scale_diff_all.png")

# # # # --- 执行部分 ---
# # # if __name__ == "__main__":
# # #     # 请在这里修改为你的实际文件名
# # #     # 假设你上传的第一个文件是 'report1.txt'，第二个是 'report2.txt'
    
# # #     report_file_2 = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k/Final_Full_Report-all-16.5.txt'  # 你的第一份报告
# # #     report_file_1 = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-8k/Final_Full_Report.txt'  # 你的第二份报告 (测试时可用同一个)
    
# # #     # 如果你有第二份文件，请取消下面这行的注释并修改文件名
# # #     # report_file_2 = 'Your_Second_Report.txt' 

# # #     compare_and_plot(report_file_1, report_file_2)

# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import matplotlib.patches as mpatches
# # import os

# # # --- 设置绘图风格 ---
# # # 尝试设置中文字体，如果乱码可根据系统调整（如 SimHei, Microsoft YaHei 等）
# # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] 
# # plt.rcParams['axes.unicode_minus'] = False
# # sns.set_style("whitegrid")

# # def parse_section_data(content, section_title, headers):
# #     """ 解析指定章节的数据 """
# #     start_idx = content.find(section_title)
# #     if start_idx == -1: return pd.DataFrame(columns=headers)
    
# #     section_content = content[start_idx:]
# #     lines = section_content.split('\n')
# #     data = []
# #     parsing = False
# #     separator_count = 0
    
# #     for line in lines:
# #         stripped = line.strip()
# #         if stripped.startswith('>>>') and section_title not in stripped: break
# #         if not stripped: continue
# #         if '----' in stripped:
# #             separator_count += 1
# #             if separator_count == 2: parsing = True
# #             continue
            
# #         if parsing:
# #             parts = [p.strip() for p in stripped.split('|') if p.strip()]
# #             if len(parts) >= len(headers) - 1:
# #                 row = parts[:len(headers)]
# #                 if len(row) < len(headers): row.append("")
# #                 data.append(row)

# #     df = pd.DataFrame(data, columns=headers)
# #     for col in df.columns:
# #         if col not in ['Scene Name', 'Cat', 'Interp']:
# #             df[col] = pd.to_numeric(df[col], errors='coerce')
# #     return df

# # def parse_report_file(filepath):
# #     """ 读取报告文件 """
# #     if not os.path.exists(filepath):
# #         print(f"File not found: {filepath}")
# #         return None, None
# #     with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
    
# #     fov_headers = ['Scene Name', 'Pred FOV', 'GT FOV', 'FOV Err', 'Pred Focal', 'GT Focal', 'Cat']
# #     scale_headers = ['Scene Name', 'Ratio', 'Std', 'Interp']
    
# #     return parse_section_data(content, "Intrinsics Detail Report", fov_headers), \
# #            parse_section_data(content, "Scale Drift Analysis", scale_headers)

# # def plot_dumbbell(data, value_col1, value_col2, label_col, title, xlabel, output_name, top_n=20):
# #     """
# #     绘制哑铃图/箭头图
# #     data: DataFrame
# #     value_col1: 旧值列名 (起点)
# #     value_col2: 新值列名 (终点)
# #     label_col: y轴标签列名 (场景名)
# #     """
# #     if data.empty: return

# #     plt.figure(figsize=(12, max(8, top_n * 0.4))) # 根据条目数量动态调整高度
    
# #     # 绘制连接线
# #     # 遍历每一行，画线
# #     for i, row in data.iterrows():
# #         color = 'green' if row[value_col2] < row[value_col1] else 'red' # 变小(变好)用绿，变大(变坏)用红
# #         # 如果是Scale Ratio，越接近1越好，逻辑稍微不同，这里暂时按数值大小变色，后面专门处理Scale
        
# #         plt.plot([row[value_col1], row[value_col2]], [i, i], color=color, alpha=0.6, linewidth=2, zorder=1)
        
# #         # 可以在终点画个箭头或者标记
# #         # plt.arrow(...) # 箭头比较难调，用散点代替

# #     # 绘制起点（Report 1）
# #     plt.scatter(data[value_col1], range(len(data)), color='grey', alpha=0.7, s=80, label='Old (Report 1)', zorder=2)
    
# #     # 绘制终点（Report 2）
# #     colors = ['green' if r[value_col2] < r[value_col1] else 'red' for _, r in data.iterrows()]
# #     plt.scatter(data[value_col2], range(len(data)), color=colors, s=80, label='New (Report 2)', zorder=3)

# #     plt.yticks(range(len(data)), data[label_col])
# #     plt.title(title, fontsize=14)
# #     plt.xlabel(xlabel)
    
# #     # 手动图例
# #     gray_patch = mpatches.Patch(color='grey', label='Report 1 (Old)')
# #     green_patch = mpatches.Patch(color='green', label='Improved (New < Old)')
# #     red_patch = mpatches.Patch(color='red', label='Worsened (New > Old)')
# #     plt.legend(handles=[gray_patch, green_patch, red_patch], loc='lower right')
    
# #     plt.tight_layout()
# #     plt.savefig(output_name)
# #     print(f"Saved: {output_name}")

# # def analyze_worst_cases(file1, file2, top_n=20):
# #     fov1, scale1 = parse_report_file(file1)
# #     fov2, scale2 = parse_report_file(file2)
    
# #     if fov1 is None or fov2 is None: return

# #     # ================= FOV ERROR 分析 =================
# #     # 合并
# #     fov_merged = pd.merge(fov1[['Scene Name', 'FOV Err']], fov2[['Scene Name', 'FOV Err']], 
# #                           on='Scene Name', suffixes=('_1', '_2'))
    
# #     # 1. 筛选逻辑：按 Report 2 (New) 的误差大小排序，取前 N 个最差的
# #     #    或者：按 max(Report1, Report2) 排序，看最严重的顽疾
# #     fov_merged['Max_Err'] = fov_merged[['FOV Err_1', 'FOV Err_2']].max(axis=1)
# #     top_fov = fov_merged.sort_values('Max_Err', ascending=False).head(top_n)
    
# #     # 此时 top_fov 包含了误差最大的20个场景
# #     # 重新按 Report 2 的误差排序，为了画图好看（从大到小排列）
# #     top_fov = top_fov.sort_values('FOV Err_2', ascending=True) 

# #     print(f"\nTop {top_n} Worst FOV Error Scenes:")
# #     print(top_fov[['Scene Name', 'FOV Err_1', 'FOV Err_2']])

# #     plot_dumbbell(top_fov, 'FOV Err_1', 'FOV Err_2', 'Scene Name',
# #                   f'Top {top_n} Worst FOV Error Scenes: Before vs After',
# #                   'FOV Error (Deg)', 'top_worst_fov_dumbbell.png', top_n)

# #     # ================= SCALE RATIO 分析 =================
# #     if not scale1.empty and not scale2.empty:
# #         scale_merged = pd.merge(scale1[['Scene Name', 'Ratio']], scale2[['Scene Name', 'Ratio']], 
# #                                 on='Scene Name', suffixes=('_1', '_2'))
        
# #         # Scale 越接近 1 越好。我们需要计算“偏离度”
# #         scale_merged['Dist_1'] = abs(scale_merged['Ratio_1'] - 1.0)
# #         scale_merged['Dist_2'] = abs(scale_merged['Ratio_2'] - 1.0)
        
# #         # 筛选：按现在的偏离度（或最大的偏离度）排序
# #         scale_merged['Max_Dist'] = scale_merged[['Dist_1', 'Dist_2']].max(axis=1)
# #         top_scale = scale_merged.sort_values('Max_Dist', ascending=False).head(top_n)
        
# #         # 为了画图，按 Report 2 的数值大小排序
# #         top_scale = top_scale.sort_values('Ratio_2', ascending=True)

# #         # 针对 Scale 修改一下画图逻辑（稍微自定义一下，因为1是中心）
# #         plt.figure(figsize=(12, max(8, top_n * 0.4)))
        
# #         # 参考线 1.0
# #         plt.axvline(1.0, color='black', linestyle='--', alpha=0.3)
        
# #         for i, row in top_scale.iterrows():
# #             # 判断变好还是变坏：如果 Dist_2 < Dist_1 则变好(绿)，否则红
# #             is_improved = row['Dist_2'] < row['Dist_1']
# #             color = 'green' if is_improved else 'red'
            
# #             plt.plot([row['Ratio_1'], row['Ratio_2']], [i, i], color=color, alpha=0.6, linewidth=2, zorder=1)
        
# #         plt.scatter(top_scale['Ratio_1'], range(len(top_scale)), color='grey', alpha=0.7, s=80, label='Old', zorder=2)
        
# #         colors = ['green' if r['Dist_2'] < r['Dist_1'] else 'red' for _, r in top_scale.iterrows()]
# #         plt.scatter(top_scale['Ratio_2'], range(len(top_scale)), color=colors, s=80, label='New', zorder=3)
        
# #         plt.yticks(range(len(top_scale)), top_scale['Scene Name'])
# #         plt.title(f'Top {top_n} Worst Scale Deviations: Before vs After (Ideal=1.0)', fontsize=14)
# #         plt.xlabel('Scale Ratio')
        
# #         # 图例
# #         gray_patch = mpatches.Patch(color='grey', label='Report 1')
# #         green_patch = mpatches.Patch(color='green', label='Better (Closer to 1.0)')
# #         red_patch = mpatches.Patch(color='red', label='Worse (Further from 1.0)')
# #         plt.legend(handles=[gray_patch, green_patch, red_patch], loc='lower right')
        
# #         plt.tight_layout()
# #         plt.savefig('top_worst_scale_dumbbell.png')
# #         print("Saved: top_worst_scale_dumbbell.png")

# # if __name__ == "__main__":
# #     # 修改文件名

# import pandas as pd
# import os

# # 设置 Pandas 显示选项，确保在终端能看到完整的表格
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)

# def parse_section_data(content, section_title, headers):
#     """ 解析指定章节的数据 """
#     start_idx = content.find(section_title)
#     if start_idx == -1: return pd.DataFrame(columns=headers)
    
#     section_content = content[start_idx:]
#     lines = section_content.split('\n')
#     data = []
#     parsing = False
#     separator_count = 0
    
#     for line in lines:
#         stripped = line.strip()
#         if stripped.startswith('>>>') and section_title not in stripped: break
#         if not stripped: continue
#         if '----' in stripped:
#             separator_count += 1
#             if separator_count == 2: parsing = True
#             continue
            
#         if parsing:
#             parts = [p.strip() for p in stripped.split('|') if p.strip()]
#             if len(parts) >= len(headers) - 1:
#                 row = parts[:len(headers)]
#                 if len(row) < len(headers): row.append("")
#                 data.append(row)

#     df = pd.DataFrame(data, columns=headers)
#     for col in df.columns:
#         if col not in ['Scene Name', 'Cat', 'Interp']:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df

# def parse_report_file(filepath):
#     if not os.path.exists(filepath):
#         print(f"File not found: {filepath}")
#         return None, None
#     with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
    
#     fov_headers = ['Scene Name', 'Pred FOV', 'GT FOV', 'FOV Err', 'Pred Focal', 'GT Focal', 'Cat']
#     scale_headers = ['Scene Name', 'Ratio', 'Std', 'Interp']
    
#     return parse_section_data(content, "Intrinsics Detail Report", fov_headers), \
#            parse_section_data(content, "Scale Drift Analysis", scale_headers)

# def print_comparison_tables(file1, file2, top_n=30):
#     print(f"Loading Report 1 (Old): {file1}")
#     print(f"Loading Report 2 (New): {file2}")
    
#     fov1, scale1 = parse_report_file(file1)
#     fov2, scale2 = parse_report_file(file2)
    
#     if fov1 is None or fov2 is None: return

#     # ================= FOV ERROR 对比表 =================
#     print("\n" + "="*80)
#     print(f"【FOV Error 对比】(按新报告误差降序排列 - Top {top_n})")
#     print("Diff = New - Old (负数表示变好了/误差减小，正数表示变差了)")
#     print("="*80)
    
#     fov_merged = pd.merge(fov1[['Scene Name', 'FOV Err', 'GT FOV']], 
#                           fov2[['Scene Name', 'FOV Err']], 
#                           on='Scene Name', suffixes=('_Old', '_New'))
    
#     # 计算差异
#     fov_merged['Diff'] = fov_merged['FOV Err_New'] - fov_merged['FOV Err_Old']
    
#     # 排序：按新报告的误差绝对值大小排序（看当前最差的是哪些）
#     # 如果想按“恶化程度”排序，可以改为 sort_values('Diff', ascending=False)
#     fov_merged_sorted = fov_merged.sort_values('FOV Err_New', ascending=False).head(top_n)
    
#     # 整理显示列
#     display_fov = fov_merged_sorted[['Scene Name', 'GT FOV', 'FOV Err_Old', 'FOV Err_New', 'Diff']]
#     print(display_fov.to_string(index=False, float_format="%.2f"))

#     # ================= SCALE RATIO 对比表 =================
#     if not scale1.empty and not scale2.empty:
#         print("\n" + "="*80)
#         print(f"【Scale Ratio 对比】(按新报告偏差程度降序排列 - Top {top_n})")
#         print("理想 Ratio 为 1.0")
#         print("Dev_New = |Ratio_New - 1.0| (越小越好)")
#         print("Change  = Dev_New - Dev_Old (负数表示回归准确，正数表示偏离更远)")
#         print("="*80)
        
#         scale_merged = pd.merge(scale1[['Scene Name', 'Ratio']], 
#                                 scale2[['Scene Name', 'Ratio']], 
#                                 on='Scene Name', suffixes=('_Old', '_New'))
        
#         # 计算距离 1.0 的偏差
#         scale_merged['Dev_Old'] = (scale_merged['Ratio_Old'] - 1.0).abs()
#         scale_merged['Dev_New'] = (scale_merged['Ratio_New'] - 1.0).abs()
#         scale_merged['Change'] = scale_merged['Dev_New'] - scale_merged['Dev_Old']
        
#         # 排序：按新报告的偏差大小排序
#         scale_merged_sorted = scale_merged.sort_values('Dev_New', ascending=False).head(top_n)
        
#         display_scale = scale_merged_sorted[['Scene Name', 'Ratio_Old', 'Ratio_New', 'Dev_New', 'Change']]
#         print(display_scale.to_string(index=False, float_format="%.4f"))

# if __name__ == "__main__":
#     # 请确认此处文件名与您服务器上的文件名一致
#     report_old = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-8k/Final_Full_Report.txt'
#     report_new = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k/Final_Full_Report-all-16.5.txt'
    
#     print_comparison_tables(report_old, report_new, top_n=20)


import pandas as pd
import os
import re

# 设置显示选项，保证表格对齐和完整
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

def parse_scale_robust(filepath):
    """ 
    暴力解析 Scale 部分，兼容各种格式差异 
    """
    if not os.path.exists(filepath):
        print(f"[错误] 找不到文件: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 定位 Scale 章节
    # 尝试多种可能的标题
    markers = ["Scale Drift Analysis", "尺度偏差分析", "Scale Analysis"]
    start_idx = -1
    for m in markers:
        idx = content.find(m)
        if idx != -1:
            start_idx = idx
            break
    
    if start_idx == -1:
        print(f"[警告] 在文件中未找到 Scale 章节: {filepath}")
        return pd.DataFrame()

    lines = content[start_idx:].split('\n')
    
    data = []
    parsing = False
    separator_count = 0
    
    for line in lines:
        stripped = line.strip()
        # 遇到下一个大章节（通常是 >>> 开头）停止
        if stripped.startswith('>>>') and start_idx not in [content.find(stripped)]:
            break
            
        if not stripped: continue
        
        # 寻找分隔线 ----
        if '----' in stripped:
            separator_count += 1
            if separator_count >= 2: # 通常表头和数据之间有第二条线
                parsing = True
            continue
            
        if parsing:
            # 提取数据：假设前两列一定是 Name 和 Ratio
            # 兼容格式： park0 | 0.6884 | ...
            parts = [p.strip() for p in stripped.split('|') if p.strip()]
            
            if len(parts) >= 2:
                scene_name = parts[0]
                ratio_str = parts[1]
                
                # 尝试将 Ratio 转为 float
                try:
                    ratio = float(ratio_str)
                    data.append({'Scene Name': scene_name, 'Ratio': ratio})
                except ValueError:
                    continue # 如果第二列不是数字，可能是乱码行，跳过

    df = pd.DataFrame(data)
    print(f"成功从 {os.path.basename(filepath)} 解析出 {len(df)} 条 Scale 数据。")
    return df

def compare_scale(file1, file2):
    print(f"正在读取旧报告: {file1}")
    df1 = parse_scale_robust(file1)
    
    print(f"正在读取新报告: {file2}")
    df2 = parse_scale_robust(file2)
    
    if df1.empty or df2.empty:
        print("\n[无法对比] 其中一份文件的 Scale 数据为空，请检查文件内容。")
        return

    # 合并对比
    merged = pd.merge(df1, df2, on='Scene Name', suffixes=('_Old', '_New'))
    
    # 计算指标
    # 1. 偏差值 (Deviation): 离 1.0 差多少 (越小越好)
    merged['Dev_Old'] = (merged['Ratio_Old'] - 1.0).abs()
    merged['Dev_New'] = (merged['Ratio_New'] - 1.0).abs()
    
    # 2. 变化量 (Change): 新偏差 - 旧偏差 (负数表示变好了/回归准确)
    merged['Change'] = merged['Dev_New'] - merged['Dev_Old']
    
    # 排序：按“新报告的偏差”从大到小排，先看最不准的
    result = merged.sort_values('Dev_New', ascending=False)
    
    # 选取显示的列
    display_cols = ['Scene Name', 'Ratio_Old', 'Ratio_New', 'Dev_New', 'Change']
    
    print("\n" + "="*90)
    print("【Scale 尺度偏差对比表】")
    print("说明：")
    print("1. Ratio (比例): 理想值为 1.0000")
    print("2. Dev_New (新偏差): |New - 1.0|，数值越大表示尺度越不准")
    print("3. Change (变化): 负数(-)表示变准了，正数(+)表示变差了")
    print("="*90)
    
    # 打印前 30 行
    print(result.head(30).to_string(index=False, float_format="%.4f"))
    
    # 额外统计：变好的有多少，变差的有多少
    improved = len(result[result['Change'] < 0])
    worsened = len(result[result['Change'] > 0])
    print("\n" + "-"*30)
    print(f"统计摘要: 共 {len(result)} 个场景")
    print(f"  ✅ 变准了 (Change < 0): {improved} 个")
    print(f"  ❌ 变差了 (Change > 0): {worsened} 个")
    print(f"  ➖ 无变化 (Change = 0): {len(result) - improved - worsened} 个")

if __name__ == "__main__":
    # 请确保文件名正确
    old_report = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-8k/Final_Full_Report.txt'
    new_report = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k/Final_Full_Report-all-16.5.txt'
    
    compare_scale(old_report, new_report)
# #     report1 = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-8k/Final_Full_Report.txt'            # 旧报告
# #     report2 = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k/Final_Full_Report-all-16.5.txt'   # 新报告
# #     #     report_file_2 = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-all-122-16.5k/Final_Full_Report-all-16.5.txt'  # 你的第一份报告
# # #     report_file_1 = '/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-8k/Final_Full_Report.txt'  # 你的第二份报告 (测试时可用同一个)
# #     analyze_worst_cases(report1, report2, top_n=20)
