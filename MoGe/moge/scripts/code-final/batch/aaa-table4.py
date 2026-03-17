import pandas as pd
import numpy as np
import os
import glob

def generate_evaluation_with_fov(score_csv_path, ref_csv_dir, output_tsv="/home/szq/moge2/MoGe/moge/scripts/code-final/batch/table4/evaluation_table.txt"):
    # ================= 1. 建立 Hash -> FOV 映射 =================
    fov_mapping = {}
    
    # 查找所有 final_dataset_*.csv (对应你的 Campus, Factory, Farm, grass)
    ref_csv_files = glob.glob(os.path.join(ref_csv_dir, "final_dataset_*.csv"))
    
    if not ref_csv_files:
        print(f"❌ 错误：在 {ref_csv_dir} 目录下没有找到 final_dataset_*.csv 文件！")
        return

    for ref_csv in ref_csv_files:
        try:
            # 读取参考表，尝试 utf-8 或 gbk 编码
            df_ref = pd.read_csv(ref_csv, encoding='utf-8')
        except UnicodeDecodeError:
            df_ref = pd.read_csv(ref_csv, encoding='gbk')
            
        # 确认必要的列存在
        if '匹配到的参考图(Target)' in df_ref.columns and 'FOV' in df_ref.columns:
            for _, row in df_ref.iterrows():
                target_pic = str(row['匹配到的参考图(Target)'])
                # 去除 .JPG 后缀，提取纯哈希名进行匹配
                hash_name = target_pic.split('.')[0]
                fov_mapping[hash_name] = row['FOV']
                
    print(f"✅ 从参考表中提取到了 {len(fov_mapping)} 个 FOV 映射关系。")

    # ================= 2. 读取分数 CSV 并匹配 FOV =================
    if not os.path.exists(score_csv_path):
        print(f"❌ 错误：找不到分数文件 {score_csv_path}")
        return

    df = pd.read_csv(score_csv_path)
    
    # 将 FOV 匹配到分数数据中
    df['FOV'] = df['Filename'].apply(lambda x: fov_mapping.get(str(x), np.nan))
    
    missing_fov_count = df['FOV'].isna().sum()
    if missing_fov_count > 0:
        print(f"⚠️ 警告: 有 {missing_fov_count} 条数据未能匹配到 FOV。")

    # ================= 3. 数据过滤与统一场景名 =================
    scene_mapping = {
        'Cleaned_Dataset_Campus': 'Campus',
        'Cleaned_Dataset_Factory': 'Factory',
        'Cleaned_Dataset_Farm': 'Farm',
        'Cleaned_Dataset_Gress': 'Gress',
        'Cleaned_Dataset_Grass': 'Gress' 
    }
    df['Scene_Mapped'] = df['Scene'].map(scene_mapping)
    df = df.dropna(subset=['Scene_Mapped'])

    # ================= 4. 个性化就近分组逻辑 =================
    def get_closest(val, targets):
        if pd.isna(val): return np.nan
        try:
            return min(targets, key=lambda x: abs(x - float(val)))
        except:
            return np.nan

    df['Target_Pitch'] = df['Pitch'].apply(lambda x: get_closest(x, [-90, -75, -60, -45]))
    df['Target_Height'] = df['Height'].apply(lambda x: get_closest(x, [80, 120]))
    df['Target_FOV'] = df['FOV'].apply(lambda x: get_closest(x, [63, 83]))

    # ================= 5. 计算指标 =================
    scenes = ['Campus', 'Factory', 'Farm', 'Gress']
    
    def get_metrics_str(condition_col, condition_val):
        row_metrics = []
        for scene in scenes:
            subset = df[(df['Scene_Mapped'] == scene) & (df[condition_col] == condition_val)]
            if subset.empty:
                row_metrics.append("-")
            else:
                # 均值计算
                absrel = subset['AbsRel'].mean() * 100
                rmse = subset['RMSE'].mean()
                a1_10 = subset['a1.10'].mean() * 100
                a1_25 = subset['a1.25'].mean() * 100
                # 拼接成单元格文本
                cell_str = f"{absrel:.2f}  {rmse:.2f}  {a1_10:.1f}  {a1_25:.1f}"
                row_metrics.append(cell_str)
        return row_metrics

    # ================= 6. 构建最终表格内容 =================
    all_rows = []
    
    # 表头
    all_rows.append(['', ''] + scenes)
    all_rows.append(['', ''] + ['AbsRel RMSE ↓ a1.10 a1.25 ↑'] * 4)
    
    # FOV 分类
    all_rows.append(['FOV', '', '', '', '', ''])
    for val, name in [(63, '63'), (83, '83')]:
        all_rows.append(['MoGe2-Ours', name] + get_metrics_str('Target_FOV', val))

    # Pitch 分类
    all_rows.append(['pitch', '', '', '', '', ''])
    for val, name in [(-90, '-90-Nadir'), (-75, '-75'), (-60, '-60'), (-45, '-45-Oblique')]:
        all_rows.append(['MoGe2-Ours', name] + get_metrics_str('Target_Pitch', val))

    # Altitude 分类
    all_rows.append(['Altitude', '', '', '', '', ''])
    for val, name in [(80, '80m'), (120, '120m')]:
        all_rows.append(['MoGe2-Ours', name] + get_metrics_str('Target_Height', val))

    # ================= 7. 输出 =================
    with open(output_tsv, 'w', encoding='utf-8') as f:
        for row in all_rows:
            f.write('\t'.join(row) + '\n')
            
    print(f"\n✅ 评估报告已生成：{output_tsv}")
    print("\n--- 直接复制下方内容粘贴到 Excel ---")
    for row in all_rows:
        print('\t'.join(row))

if __name__ == "__main__":
    # 分数的绝对路径
    score_csv_file = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed.csv" 
    
    # 参考表的绝对路径
    ref_dir = "/data1/szq/Val/Bench"
    
    generate_evaluation_with_fov(score_csv_path=score_csv_file, ref_csv_dir=ref_dir)