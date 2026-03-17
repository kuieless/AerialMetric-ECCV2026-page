import pandas as pd
import numpy as np
import os
import glob

def update_evaluation_dataset(score_csv_path, ref_csv_dir, output_csv_path):
    print("🚀 开始处理数据...")
    
    # ================= 1. 建立 Hash -> FOV 映射 =================
    fov_mapping = {}
    ref_csv_files = glob.glob(os.path.join(ref_csv_dir, "final_dataset_*.csv"))
    
    if not ref_csv_files:
        print(f"❌ 错误：在 {ref_csv_dir} 目录下没有找到 final_dataset_*.csv 文件！")
        return

    for ref_csv in ref_csv_files:
        try:
            df_ref = pd.read_csv(ref_csv, encoding='utf-8')
        except UnicodeDecodeError:
            df_ref = pd.read_csv(ref_csv, encoding='gbk')
            
        if '匹配到的参考图(Target)' in df_ref.columns and 'FOV' in df_ref.columns:
            for _, row in df_ref.iterrows():
                target_pic = str(row['匹配到的参考图(Target)'])
                # 去除后缀提取纯哈希名
                hash_name = target_pic.split('.')[0]
                fov_mapping[hash_name] = row['FOV']
                
    print(f"✅ 成功从参考表中提取到了 {len(fov_mapping)} 个 FOV 映射关系。")

    # ================= 2. 读取分数 CSV 并匹配 FOV =================
    if not os.path.exists(score_csv_path):
        print(f"❌ 错误：找不到分数文件 {score_csv_path}")
        return

    df = pd.read_csv(score_csv_path)
    initial_row_count = len(df)
    print(f"📄 读取到分数表，共 {initial_row_count} 行数据。")
    
    # 提取分数表中的 Filename 哈希并匹配 FOV
    # 假设 score 表里的 Filename 也是哈希值（以防带有后缀，统一用 split 处理）
    df['FOV'] = df['Filename'].apply(lambda x: fov_mapping.get(str(x).split('.')[0], np.nan))
    
    missing_fov_count = df['FOV'].isna().sum()
    if missing_fov_count > 0:
        print(f"⚠️ 警告: 有 {missing_fov_count} 条数据未能匹配到 FOV（保留为 NaN）。")

    # ================= 3. 个性化就近分组逻辑 (订正 Group) =================
    def get_closest(val, targets):
        """找到距离 val 最近的 target 值"""
        if pd.isna(val): 
            return np.nan
        try:
            return min(targets, key=lambda x: abs(x - float(val)))
        except:
            return np.nan

    # 定义各个参数的目标分组范围 (根据你之前的代码逻辑设定)
    pitch_targets = [-90, -75, -60, -45]
    height_targets = [80, 120]
    fov_targets = [63, 83]

    print("🔄 正在根据就近原则重新划分 Pitch_Group, Height_Group 和 FOV_Group...")
    # 覆盖原有的错误分组或创建新分组列
    df['Pitch_Group'] = df['Pitch'].apply(lambda x: get_closest(x, pitch_targets))
    df['Height_Group'] = df['Height'].apply(lambda x: get_closest(x, height_targets))
    df['FOV_Group'] = df['FOV'].apply(lambda x: get_closest(x, fov_targets))

    # ================= 4. 整理列顺序并导出 =================
    # 将新增的 FOV 和 FOV_Group 放到靠前的位置，方便查看
    cols = df.columns.tolist()
    # 把我们关心的列提到前面，其余列跟在后面
    priority_cols = ['Scene', 'Filename', 'Pitch', 'Pitch_Group', 'Height', 'Height_Group', 'FOV', 'FOV_Group']
    remaining_cols = [c for c in cols if c not in priority_cols]
    final_cols = priority_cols + remaining_cols
    
    df = df[final_cols]

    # 保存为新的 CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"🎉 处理完成！包含 FOV 和正确分组的新表格已保存至：\n📂 {output_csv_path}")


if __name__ == "__main__":
    # 原始分数的绝对路径
    score_csv_file = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed.csv" 
    # score_csv_file = "/data1/szq/Infer-Final/Table2-1/Table2-head-LoRA-96-groundless4-UElr2/00001200/Extracted/Bench/Eval_Report_Bench_Detailed.csv" 

    # 参考表的绝对路径 (包含 final_dataset_*.csv 的目录)
    ref_dir = "/data1/szq/Val/Bench"
    
    # 导出的新表路径 (建议加个后缀以防覆盖原文件)
    output_csv_file = "/data1/szq/Table2-final-baseline/vitl-normal/Extracted/Bench/Eval_Report_Bench_Detailed_with_FOV-baseline.csv"
    
    update_evaluation_dataset(
        score_csv_path=score_csv_file, 
        ref_csv_dir=ref_dir,
        output_csv_path=output_csv_file
    )