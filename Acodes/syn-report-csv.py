import os
import glob
import pandas as pd
import re

# ================= 配置区域 =================
# 你的日志文件夹路径
LOG_DIR = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Logs-origin"

# 输出的汇总表格路径
OUTPUT_CSV = os.path.join(LOG_DIR, "originFinal_Analysis_Summary.csv")
# ===========================================

def parse_report_file(filepath):
    """解析单个 TXT 报告文件"""
    filename = os.path.basename(filepath)
    # 从文件名提取场景名: Report_Batch_City.txt -> City
    scene_name = filename.replace("Report_Batch_", "").replace(".txt", "")
    
    with open(filepath, 'r') as f:
        content = f.read()

    # 存储提取到的数据
    extracted_data = []

    # 定义我们要寻找的两个大任务块
    task_map = {
        "Absolute_Metric": "Absolute (No Scale)",
        "Relative_Structure": "Relative (Median Scale)"
    }

    # 定义我们要提取的所有行关键词
    # 修正点：Pitch 部分必须补全 ("Pitch", "关键词")
    targets = [
        # Overall 通常是 Pandas 输出的 index 0
        ("Overall", "0      "), 
        
        # FOV
        ("FOV", "FOV_64"), ("FOV", "FOV_74"), ("FOV", "FOV_84"),
        
        # Altitude
        ("Altitude", "Alt_Low"), ("Altitude", "Alt_Mid"), ("Altitude", "Alt_High"),
        
        # Pitch (这里修正了)
        ("Pitch", "Pitch_-90"), 
        ("Pitch", "Pitch_-75"), 
        ("Pitch", "Pitch_-60"), 
        ("Pitch", "Pitch_-45"), 
        ("Pitch", "Pitch_-30")
    ]

    # 1. 按任务分割文本
    for task_key, task_display_name in task_map.items():
        # 寻找 >>> TASK: Absolute_Metric <<<
        start_marker = f">>> TASK: {task_key} <<<"
        if start_marker not in content:
            continue
            
        # 截取该任务之后的文本，直到下一个任务开始或文件结束
        task_content = content.split(start_marker)[1]
        # 如果后面还有其他任务，截断它
        if ">>> TASK:" in task_content:
            task_content = task_content.split(">>> TASK:")[0]
        
        lines = task_content.split('\n')
        
        # 2. 逐行扫描目标数据
        for dim, keyword in targets:
            for line in lines:
                # 检查行是否包含关键词 (且关键词在行首附近，避免误匹配)
                clean_line = line.strip()
                
                # 兼容 Pandas 不同版本的空格输出 (可能有多个空格)
                if clean_line.startswith(keyword) or (keyword.startswith("0") and clean_line.startswith("0 ")):
                    # 解析数字
                    parts = clean_line.split()
                    
                    # 尝试找到最后三个数字 (AbsRel, RMSE, a1)
                    numeric_values = []
                    for p in parts:
                        try:
                            val = float(p)
                            numeric_values.append(val)
                        except:
                            continue
                    
                    # 我们只需要最后三个: AbsRel, RMSE, a1
                    if len(numeric_values) >= 3:
                        metrics = numeric_values[-3:]
                        
                        # 特殊处理 Overall 的行名
                        row_name = keyword.strip()
                        if row_name.startswith("0"): row_name = "Overall"
                        
                        extracted_data.append({
                            "Scene": scene_name,
                            "Task": task_display_name,
                            "Dimension": dim,
                            "Group": row_name,
                            "AbsRel": metrics[0],
                            "RMSE": metrics[1],
                            "a1": metrics[2]
                        })
                    break # 找到后跳出当前 keyword 循环，找下一个 keyword

    return extracted_data

def main():
    print(f"📂 Scanning logs in: {LOG_DIR}")
    report_files = glob.glob(os.path.join(LOG_DIR, "Report_Batch_*.txt"))
    
    if not report_files:
        print("❌ No report files found!")
        return

    all_data = []
    for f in report_files:
        print(f"   -> Parsing: {os.path.basename(f)}")
        try:
            data = parse_report_file(f)
            all_data.extend(data)
        except Exception as e:
            print(f"      ⚠️ Error parsing {f}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        
        # 调整列顺序
        cols = ["Scene", "Task", "Dimension", "Group", "AbsRel", "RMSE", "a1"]
        df = df[cols]
        
        # 排序
        df = df.sort_values(by=["Scene", "Task", "Dimension", "Group"])
        
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*50)
        print(f"✅ Summary saved to: {OUTPUT_CSV}")
        print("="*50)
        print("Data Preview:")
        print(df.head(10).to_string())
    else:
        print("❌ No valid data extracted.")

if __name__ == "__main__":
    main()