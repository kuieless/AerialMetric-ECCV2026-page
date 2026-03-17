import os
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 如果没安装 tqdm，可以去掉相关代码或 pip install tqdm

# ================= 配置区域 =================
# 数据集根目录
ROOT_DIR = r"/data1/szq/TrainingData_Final_MoGe-All"

# 结果保存文件
OUTPUT_TXT = "all_black_samples.txt"

# 是否要把检测到的坏文件直接移动到回收站？(建议先 False，检查 txt 没问题再改成 True)
MOVE_TO_TRASH = False 
TRASH_DIR = os.path.join(ROOT_DIR, "_TRASH_ZEROS")
# ===========================================

def check_file(file_info):
    """检查单个文件是否全为0"""
    depth_path, folder_path = file_info
    
    try:
        # 1. 加载数据 (只读取不解压，速度快)
        # allow_pickle=True 防止报错
        data = np.load(depth_path, allow_pickle=True)
        
        # 2. 简单的维度清洗
        if data.ndim == 3: 
            data = np.squeeze(data)
        
        # 3. 核心判断：是否全是 0 (或者小于等于0)
        # 只要最大值小于等于0，或者所有值都是0，就认为是全黑
        # 使用 np.max 读取头部极大值，比 np.all 快一点点
        # 也可以用 nan_to_num 防止 nan 报错
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.max(data) <= 1e-5:  # 容差，防止极小的浮点数噪音
            return folder_path
            
    except Exception as e:
        print(f"\n[Error] {depth_path}: {e}")
        return None
    
    return None

def main():
    print(f"🚀 开始扫描根目录: {ROOT_DIR}")
    
    # 1. 快速搜集所有 depth.npy 路径
    all_depth_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        if "_TRASH" in root: continue # 跳过垃圾箱
        if "depth.npy" in files:
            all_depth_files.append((os.path.join(root, "depth.npy"), root))
            
    print(f"📂 找到 {len(all_depth_files)} 个深度图文件，开始校验数值...")

    bad_samples = []
    
    # 2. 多线程并发检查 (加快 IO 读取速度)
    # 你的电脑应该能轻松跑 16-32 个线程
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(check_file, all_depth_files), total=len(all_depth_files), unit="img"))
        
    # 过滤掉 None
    bad_samples = [r for r in results if r is not None]

    print("\n" + "="*40)
    print(f"📊 扫描完成！")
    print(f"❌ 全黑/全0 样本数量: {len(bad_samples)}")
    print("="*40)

    # 3. 保存结果到 TXT
    if bad_samples:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            for path in bad_samples:
                f.write(path + "\n")
        print(f"📝 坏样本路径已保存至: {os.path.abspath(OUTPUT_TXT)}")
        
        # 4. (可选) 自动移动
        if MOVE_TO_TRASH:
            print(f"\n🗑️ 正在移动到回收站: {TRASH_DIR} ...")
            if not os.path.exists(TRASH_DIR): os.makedirs(TRASH_DIR)
            
            count = 0
            for folder in bad_samples:
                try:
                    folder_name = os.path.basename(folder)
                    # 父文件夹名做前缀，防止重名
                    parent_name = os.path.basename(os.path.dirname(folder)) 
                    target_name = f"{parent_name}_{folder_name}"
                    target_path = os.path.join(TRASH_DIR, target_name)
                    
                    shutil.move(folder, target_path)
                    count += 1
                except Exception as e:
                    print(f"移动失败 {folder}: {e}")
            print(f"✅ 成功移动 {count} 个文件夹。")
    else:
        print("🎉 恭喜！没有发现全为 0 的坏数据。")

if __name__ == "__main__":
    main()