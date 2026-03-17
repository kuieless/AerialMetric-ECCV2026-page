# import pandas as pd
# import numpy as np
# import os
# import glob

# def update_evaluation_dataset(score_txt_path, ref_csv_dir, output_csv_path, output_scene_stats, output_param_stats):
#     print("🚀 开始处理数据...")
    
#     # ================= 1. 建立 Hash -> 元数据映射 =================
#     metadata_mapping = {}
#     ref_csv_files = glob.glob(os.path.join(ref_csv_dir, "final_dataset_*.csv"))
    
#     if not ref_csv_files:
#         print(f"❌ 错误：在 {ref_csv_dir} 目录下没有找到 final_dataset_*.csv 文件！")
#         return

#     for ref_csv in ref_csv_files:
#         try:
#             df_ref = pd.read_csv(ref_csv, encoding='utf-8')
#         except UnicodeDecodeError:
#             df_ref = pd.read_csv(ref_csv, encoding='gbk')
            
#         if '匹配到的参考图(Target)' in df_ref.columns:
#             for _, row in df_ref.iterrows():
#                 target_pic = str(row['匹配到的参考图(Target)'])
#                 # 防止遇到空行报错
#                 if pd.isna(target_pic) or target_pic.strip() == '':
#                     continue
                    
#                 hash_name = target_pic.split('.')[0].split('/')[-1]
                
#                 # ⚠️ 关键修改点：根据你提供的真实表头提取数据
#                 # 如果有'实际Pitch'就用实际的进行就近测算，没有就用'Pitch分类'
#                 metadata_mapping[hash_name] = {
#                     'Scene': row.get('源文件夹', 'Unknown'),  
#                     'FOV': row.get('FOV', np.nan),
#                     'Pitch': row.get('实际Pitch', row.get('Pitch分类', np.nan)), 
#                     'Altitude': row.get('相对高度(自动)', np.nan) 
#                 }
                
#     print(f"✅ 成功从参考表中提取到了 {len(metadata_mapping)} 个图像的元数据。")

#     # ================= 2. 读取新的 txt 分数文件 =================
#     if not os.path.exists(score_txt_path):
#         print(f"❌ 错误：找不到分数文件 {score_txt_path}")
#         return

#     # 使用 \t 制表符分割
#     df = pd.read_csv(score_txt_path, sep='\t')
#     print(f"📄 读取到分数表，共 {len(df)} 行数据。")
    
#     # 提取纯哈希值
#     df['Hash'] = df['filename'].apply(lambda x: str(x).split('/')[-1].split('.')[0])
    
#     # 映射元数据
#     df['Scene'] = df['Hash'].apply(lambda x: metadata_mapping.get(x, {}).get('Scene', 'Unknown'))
#     df['FOV'] = df['Hash'].apply(lambda x: metadata_mapping.get(x, {}).get('FOV', np.nan))
#     df['Pitch'] = df['Hash'].apply(lambda x: metadata_mapping.get(x, {}).get('Pitch', np.nan))
#     df['Altitude'] = df['Hash'].apply(lambda x: metadata_mapping.get(x, {}).get('Altitude', np.nan))

#     # ================= 3. 就近分组逻辑 (你要求的划组方式) =================
#     def get_closest(val, targets):
#         """找到距离 val 最近的 target 值，比如 120.1 归为 120"""
#         if pd.isna(val) or str(val).strip() == '': 
#             return np.nan
#         try:
#             return min(targets, key=lambda x: abs(x - float(val)))
#         except:
#             return np.nan

#     # 严格按照你定义的目标分组
#     pitch_targets = [-90, -75, -60, -45]
#     altitude_targets = [80, 120]  
#     fov_targets = [63, 83]

#     df['Pitch_Group'] = df['Pitch'].apply(lambda x: get_closest(x, pitch_targets))
#     df['Altitude_Group'] = df['Altitude'].apply(lambda x: get_closest(x, altitude_targets))
#     df['FOV_Group'] = df['FOV'].apply(lambda x: get_closest(x, fov_targets))

#     # 保存明细表
#     df.to_csv(output_csv_path, index=False, encoding='utf-8')
#     print(f"🎉 明细表导出成功！包含正确对齐的分数和元数据：\n📂 {output_csv_path}")

#     # ================= 4. 统计 1：按场景 (Scene) 分组统计 =================
#     print("\n📊 正在统计每个 Scene 下的评估指标...")
#     target_metrics = ['depth_metric.absrel', 'depth_metric.rmse', 'depth_metric.delta1']
#     available_metrics = [m for m in target_metrics if m in df.columns]
    
#     if available_metrics:
#         scene_stats = df.groupby('Scene')[available_metrics].mean().reset_index()
#         scene_counts = df.groupby('Scene').size().reset_index(name='Sample_Count')
#         scene_stats = pd.merge(scene_stats, scene_counts, on='Scene')
        
#         scene_stats.to_csv(output_scene_stats, index=False, encoding='utf-8')
#         print(f"📈 场景统计完成！已保存至：\n📂 {output_scene_stats}")
    
#     # ================= 5. 统计 2：按三大参数组合分组统计 =================
#     print("\n📊 正在统计按 FOV, Pitch, Altitude 分组的评估指标...")
#     # 过滤掉缺失三大参数任意一个的数据
#     valid_param_df = df.dropna(subset=['Pitch_Group', 'Altitude_Group', 'FOV_Group'])
    
#     if len(valid_param_df) == 0:
#         print("⚠️ 警告：因为提取不到参数，无法生成参数分组统计表！请检查表头映射。")
#     elif available_metrics:
#         # 按照 FOV -> Pitch -> Altitude 的层级组合进行计算均值
#         param_stats = valid_param_df.groupby(['FOV_Group', 'Pitch_Group', 'Altitude_Group'])[available_metrics].mean().reset_index()
#         param_counts = valid_param_df.groupby(['FOV_Group', 'Pitch_Group', 'Altitude_Group']).size().reset_index(name='Sample_Count')
#         param_stats = pd.merge(param_stats, param_counts, on=['FOV_Group', 'Pitch_Group', 'Altitude_Group'])
        
#         param_stats.to_csv(output_param_stats, index=False, encoding='utf-8')
#         print(f"📈 参数分组统计完成！已保存至：\n📂 {output_param_stats}")
#         print(param_stats.head().to_string(index=False))


# if __name__ == "__main__":
#     score_txt_file = "/data1/yzy/Benchmark/MoGe/eval_output/unidepth_v2_vitl14_Bench_per_image.txt" 
#     ref_dir = "/data1/szq/Val/Bench-ori"
    
#     output_csv_file = "/data1/szq/Val/Bench-ori/eval_output/unidepth_v2_vitl14_Detailed_with_FOV.csv"
#     output_scene_file = "/data1/szq/Val/Bench-ori/eval_output/unidepth_v2_vitl14_Stats_by_Scene.csv"
#     output_param_file = "/data1/szq/Val/Bench-ori/eval_output/unidepth_v2_vitl14_Stats_by_Params.csv"
    
#     update_evaluation_dataset(
#         score_txt_path=score_txt_file, 
#         ref_csv_dir=ref_dir,
#         output_csv_path=output_csv_file,
#         output_scene_stats=output_scene_file,
#         output_param_stats=output_param_file
#     )

import pandas as pd
import numpy as np
import os
import glob

def generate_evaluation_report(eval_txt_path, ref_csv_dir, output_tsv_path):
    print("🚀 开始处理评估数据...")

    # ================= 1. 从参考表提取 Hash -> [FOV, Pitch, Altitude] 映射 =================
    ref_mapping = {}
    ref_csv_files = glob.glob(os.path.join(ref_csv_dir, "final_dataset_*.csv"))
    
    if not ref_csv_files:
        print(f"❌ 错误：在 {ref_csv_dir} 目录下没有找到 final_dataset_*.csv 文件！")
        return

    for ref_csv in ref_csv_files:
        try:
            df_ref = pd.read_csv(ref_csv, encoding='utf-8')
        except UnicodeDecodeError:
            df_ref = pd.read_csv(ref_csv, encoding='gbk')
            
        # 确认必要的列存在
        required_cols = ['匹配到的参考图(Target)', 'FOV', '相对高度(自动)']
        if all(col in df_ref.columns for col in required_cols):
            for _, row in df_ref.iterrows():
                target_pic = str(row['匹配到的参考图(Target)'])
                hash_name = target_pic.split('.')[0]  # 去除 .JPG 后缀
                
                # 优先取实际Pitch，如果没有则取Pitch分类
                pitch_val = row.get('实际Pitch', row.get('Pitch分类', np.nan))
                
                ref_mapping[hash_name] = {
                    'FOV': row['FOV'],
                    'Pitch': pitch_val,
                    'Altitude': row['相对高度(自动)']
                }
                
    print(f"✅ 从参考表中提取到了 {len(ref_mapping)} 个有效映射关系。")

    # ================= 2. 读取新的评估 TXT 文件 =================
    if not os.path.exists(eval_txt_path):
        print(f"❌ 错误：找不到评估文件 {eval_txt_path}")
        return

    # 注意：根据你提供的格式，这是一个以制表符分隔的文件
    df_eval = pd.read_csv(eval_txt_path, sep='\t')
    print(f"📄 读取到评估表，共 {len(df_eval)} 行数据。")

    # 过滤掉非 ok 的数据（如果需要的话）
    df_eval = df_eval[df_eval['status'] == 'ok']

    # 拆分 filename 列，提取 Scene 和纯 Hash
    # 例如：Cleaned_Dataset_Campus/003d14f8... -> ['Cleaned_Dataset_Campus', '003d14f8...']
    split_info = df_eval['filename'].str.split('/', n=1, expand=True)
    df_eval['Scene_Raw'] = split_info[0]
    df_eval['Hash'] = split_info[1]

    # 映射到标准场景名
    scene_mapping = {
        'Cleaned_Dataset_Campus': 'Campus',
        'Cleaned_Dataset_Factory': 'Factory',
        'Cleaned_Dataset_Farm': 'Farm',
        'Cleaned_Dataset_Gress': 'Gress',
        'Cleaned_Dataset_Grass': 'Gress' 
    }
    df_eval['Scene'] = df_eval['Scene_Raw'].map(scene_mapping)
    df_eval = df_eval.dropna(subset=['Scene'])  # 丢弃无法识别场景的数据

    # ================= 3. 将参考属性匹配到评估数据中 =================
    df_eval['FOV'] = df_eval['Hash'].apply(lambda x: ref_mapping.get(str(x), {}).get('FOV', np.nan))
    df_eval['Pitch'] = df_eval['Hash'].apply(lambda x: ref_mapping.get(str(x), {}).get('Pitch', np.nan))
    df_eval['Altitude'] = df_eval['Hash'].apply(lambda x: ref_mapping.get(str(x), {}).get('Altitude', np.nan))

    # ================= 4. 个性化就近分组逻辑 =================
    def get_closest(val, targets):
        if pd.isna(val): return np.nan
        try:
            return min(targets, key=lambda x: abs(x - float(val)))
        except:
            return np.nan

    # 执行就近靠近分配
    df_eval['Target_Pitch'] = df_eval['Pitch'].apply(lambda x: get_closest(x, [-90, -75, -60, -45]))
    df_eval['Target_Height'] = df_eval['Altitude'].apply(lambda x: get_closest(x, [80, 120]))
    df_eval['Target_FOV'] = df_eval['FOV'].apply(lambda x: get_closest(x, [63, 83]))

    # ================= 5. 计算指标与组装表格 =================
    scenes = ['Campus', 'Factory', 'Farm', 'Gress']
    
    def get_metrics_str(condition_col, condition_val):
        row_metrics = []
        for scene in scenes:
            subset = df_eval[(df_eval['Scene'] == scene) & (df_eval[condition_col] == condition_val)]
            if subset.empty:
                row_metrics.append("-")
            else:
                # 均值计算 (依据之前代码习惯，误差相关乘100展示更好看，RMSE保持原值)
                absrel = subset['depth_metric.absrel'].mean() * 100
                rmse = subset['depth_metric.rmse'].mean()
                delta1 = subset['depth_metric.delta1'].mean() * 100
                
                # 拼接成单元格文本
                cell_str = f"{absrel:.2f}  {rmse:.2f}  {delta1:.1f}"
                row_metrics.append(cell_str)
        return row_metrics

    # 构建最终表格内容
    all_rows = []
    
    # 表头
    all_rows.append(['', ''] + scenes)
    all_rows.append(['', ''] + ['AbsRel RMSE ↓ Delta1 ↑'] * 4) # 适配新的指标名
    
    # FOV 分类
    all_rows.append(['FOV', '', '', '', '', ''])
    for val, name in [(63, '63'), (83, '83')]:
        all_rows.append(['Model-Ours', name] + get_metrics_str('Target_FOV', val))

    # Pitch 分类
    all_rows.append(['pitch', '', '', '', '', ''])
    for val, name in [(-90, '-90-Nadir'), (-75, '-75'), (-60, '-60'), (-45, '-45-Oblique')]:
        all_rows.append(['Model-Ours', name] + get_metrics_str('Target_Pitch', val))

    # Altitude 分类
    all_rows.append(['Altitude', '', '', '', '', ''])
    for val, name in [(80, '80m'), (120, '120m')]:
        all_rows.append(['Model-Ours', name] + get_metrics_str('Target_Height', val))

    # ================= 6. 导出最终统计报告 =================
    os.makedirs(os.path.dirname(output_tsv_path), exist_ok=True)
    with open(output_tsv_path, 'w', encoding='utf-8') as f:
        for row in all_rows:
            f.write('\t'.join(row) + '\n')
            
    print(f"\n✅ 评估报告已生成：{output_tsv_path}")
    print("\n--- 直接复制下方内容粘贴到 Excel ---")
    for row in all_rows:
        print('\t'.join(row))

if __name__ == "__main__":
    # 新格式结果 TXT 文件的绝对路径 (请替换为实际路径)
    eval_txt_file = "/data1/szq/Val/Bench-ori/uni.txt" 
    
    # 参考表存放的目录绝对路径
    ref_dir = "/data1/szq/Val/Bench-ori"
    
    # 输出的汇总表格路径
    output_table_file = "/data1/szq/Val/Bench-ori/evaluation_table_updated.txt"
    
    generate_evaluation_report(
        eval_txt_path=eval_txt_file, 
        ref_csv_dir=ref_dir, 
        output_tsv_path=output_table_file
    )