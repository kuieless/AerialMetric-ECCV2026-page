import os
import json

def process_json_to_txt(file_paths, output_txt="output_scores.txt"):
    # 数据集顺序
    datasets = ['mean', 'NYUv2', 'KITTI', 'ETH3D', 'iBims-1', 'DDAD', 'DIODE', 'HAMMER']
    display_names = ['Mean', 'NYUv2', 'KITTI', 'ETH3D', 'iBims', 'DDAD', 'DIODE', 'HAMMER']
    
    # 设置列宽
    col_width = 22       # 数据列的宽度
    first_col_width = 65 # 调宽第一列，因为你的实验文件夹名字比较长
    
    # 构建表头
    header1 = f"{'Experiment':<{first_col_width}}" + "".join([f"{name:<{col_width}}" for name in display_names])
    header2 = f"{'':<{first_col_width}}" + "".join([f"{'AbsRel ↓  a1.25 ↑':<{col_width}}" for _ in display_names])
    separator = "-" * len(header1)
    
    all_lines = [header1, header2, separator]
    
    # 直接遍历你提供的具体文件路径列表
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件: {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取有意义的实验名 (取倒数第二和第三级目录名，例如 "Table2.../00000800")
            parts = file_path.split(os.sep)
            if len(parts) >= 3:
                exp_name = f"{parts[-3]}/{parts[-2]}"
            else:
                exp_name = os.path.basename(file_path)
            
            # 第一列填入实验名
            row_str = f"{exp_name:<{first_col_width}}"
            
            for ds in datasets:
                if ds in data and 'depth_metric' in data[ds]:
                    metrics = data[ds]['depth_metric']
                    
                    # 提取并转换分数（乘以100）
                    rel = metrics.get('rel', 0) * 100
                    delta1 = metrics.get('delta1', 0) * 100
                    
                    # 拼接并对齐
                    cell_val = f"{rel:.2f}  {delta1:.1f}"
                    row_str += f"{cell_val:<{col_width}}"
                else:
                    # 缺失数据时填 '-'
                    row_str += f"{'-':<{col_width}}"
                    
            all_lines.append(row_str)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 写入纯文本 TXT 文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
            
    # 在控制台打印预览
    print(f"✅ 处理完成！已保存为纯文本文件: {os.path.abspath(output_txt)}")
    print("\n--- TXT 文件内容预览 ---\n")
    for line in all_lines:
        print(line)
    print("\n------------------------")

# ================= 使用方法 =================
if __name__ == "__main__":
    # 在这里填入你具体的 metrics.json 路径
    files_to_process = [
        # "/data1/szq/Table2-final-head108-Ground/00010800_ema/metrics.json", 
        # "/data1/szq/Table2-head-LoRA-96-Ground-groundless4-UElr2-aerial/00000600/metrics.json", 
        "/data1/szq/Table2-head-LoRA-96-Ground-groundless4-UElr2-aerial3/00004800/metrics.json", 
     
    ]
    
    process_json_to_txt(files_to_process, output_txt="final_scores_aligned.txt")