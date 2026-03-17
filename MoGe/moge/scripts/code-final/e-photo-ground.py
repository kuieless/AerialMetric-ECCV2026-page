import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 请将此处修改为你的实际根目录路径
ROOT_DIR = "/data1/szq/Table-head-ground/"

# 你想要提取的指标类别 (根据你的描述，这里选用了 depth_affine_invariant)
# 如果你想换成 depth_metric，只需将此处改为 "depth_metric" 即可
METRIC_CATEGORY = "depth_metric" 

# 数据集顺序 (将按照这个顺序排列列)
DATASETS = ["NYUv2", "KITTI", "ETH3D", "iBims-1", "DDAD", "DIODE", "HAMMER", "mean"]

# 指标映射关系: JSON中的key -> CSV显示的名称
METRIC_MAP = {
    "rel": "AbsRel",
    "delta1": "a1.25"
}
# ===========================================

def extract_checkpoint_number(folder_name):
    """从文件夹名中提取数字 (例如 '00001200_ema' -> 1200)"""
    match = re.search(r'(\d+)_ema', folder_name)
    if match:
        return int(match.group(1))
    return -1

def main():
    # 1. 寻找所有的 metrics.json 文件
    search_pattern = os.path.join(ROOT_DIR, "*_ema", "metrics.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"在 {ROOT_DIR} 下未找到任何 metrics.json 文件，请检查路径。")
        return

    data_records = []

    print(f"找到 {len(files)} 个评估文件，开始解析...")

    # 2. 遍历文件提取数据
    for file_path in files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        ckpt_num = extract_checkpoint_number(folder_name)
        
        if ckpt_num == -1:
            continue

        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
        except Exception as e:
            print(f"读取 {file_path} 失败: {e}")
            continue

        # 准备当前 Checkpoint 的一行数据
        row = {'Checkpoint': ckpt_num}
        
        for dataset in DATASETS:
            # 检查数据集是否存在于 json 中
            if dataset in content and METRIC_CATEGORY in content[dataset]:
                metrics = content[dataset][METRIC_CATEGORY]
                
                # 提取 AbsRel (rel)
                if 'rel' in metrics:
                    # 注意：通常 AbsRel 显示为百分比或原值，这里保留原值，画图时可调整
                    row[(dataset, 'AbsRel')] = metrics['rel']
                
                # 提取 a1.25 (delta1)
                if 'delta1' in metrics:
                    # 通常 delta1 显示为 0-1 之间的小数，有些习惯 *100
                    row[(dataset, 'a1.25')] = metrics['delta1']
            else:
                # 如果缺少数据，填充 None
                row[(dataset, 'AbsRel')] = None
                row[(dataset, 'a1.25')] = None
        
        data_records.append(row)

    # 3. 构建 DataFrame 并排序
    df = pd.DataFrame(data_records)
    df = df.set_index('Checkpoint')
    df = df.sort_index()

    # 4. 重新组织列名以符合你的 CSV 要求 (Dataset 在上, Metric 在下)
    # 创建 MultiIndex Columns
    # 目前 df 的列是 tuple 形式: ('NYUv2', 'AbsRel'), ...
    # 我们需要确保列的顺序符合 DATASETS 的顺序
    new_columns = []
    for ds in DATASETS:
        new_columns.append((ds, 'AbsRel'))
        new_columns.append((ds, 'a1.25'))
    
    # 筛选并重新排序存在的列
    existing_columns = [c for c in new_columns if c in df.columns]
    df = df[existing_columns]
    
    # 设置 MultiIndex 表头
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Dataset', 'Metric'])

    # 5. 导出 CSV
    output_csv = "evaluation_summary.csv"
    # float_format='%.4f' 保留4位小数，你可以根据需要改成 %.2f
    df.to_csv(output_csv, float_format='%.4f')
    print(f"\nCSV 汇总表已生成: {output_csv}")
    print("-" * 30)
    print(df.head()) # 打印前几行预览

    # 6. 绘图
    plot_metrics(df)

def plot_metrics(df):
    """绘制 AbsRel 和 a1.25 的变化曲线"""
    checkpoints = df.index
    datasets = df.columns.levels[0] # 获取所有数据集名称

    # --- 图 1: AbsRel ---
    plt.figure(figsize=(12, 6))
    for ds in datasets:
        if (ds, 'AbsRel') in df.columns:
            # 获取数据，处理可能的缺失值
            series = df[(ds, 'AbsRel')].dropna()
            plt.plot(series.index, series.values, marker='o', markersize=4, label=ds)
    
    plt.title(f'AbsRel (Lower is Better) - {METRIC_CATEGORY}')
    plt.xlabel('Checkpoint')
    plt.ylabel('AbsRel')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('curve_absrel.png', dpi=300)
    print("图表已保存: curve_absrel.png")

    # --- 图 2: a1.25 (Delta1) ---
    plt.figure(figsize=(12, 6))
    for ds in datasets:
        if (ds, 'a1.25') in df.columns:
            series = df[(ds, 'a1.25')].dropna()
            plt.plot(series.index, series.values, marker='o', markersize=4, label=ds)

    plt.title(f'Delta1 / a1.25 (Higher is Better) - {METRIC_CATEGORY}')
    plt.xlabel('Checkpoint')
    plt.ylabel('Accuracy (a1 < 1.25)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('curve_accuracy.png', dpi=300)
    print("图表已保存: curve_accuracy.png")

if __name__ == "__main__":
    main()