import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from colorama import Fore, Style, init

# 初始化彩色打印
init(autoreset=True)

# ================= ⚠️ 核心配置区 ⚠️ =================

# 【重要】True = 只打印不删除 (安全模式)； False = 真正删除文件
DRY_RUN = False 

# 判定阈值：如果一张图中 0 值的比例超过 95%，则删除该对文件
INVALID_RATIO_THRESHOLD = 0.95

# 文件夹候选名
IMG_CANDIDATES = ["images_downsampled", "images", "img", "rgb"]
DEPTH_CANDIDATES = ["npy_downsampled", "npy", "depth", "depth_npy"]

# ================= 场景列表 =================
SCENE_CONFIGS = [
    # {
    #     "name": "hav",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/hav",
    #     "split": "train"
    # },
    # {
    #     "name": "lfls",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/lfls",
    #     "split": "train"
    # },
    # {
    #     "name": "lfls2",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/lfls2",
    #     "split": "train"
    # },
    # {
    #     "name": "lower",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/lower",
    #     "split": "val"
    # },
    # {
    #     "name": "SMBU",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/SMBU",
    #     "split": "train"
    # },
    # {
    #     "name": "sziit",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/sziit",
    #     "split": "val"
    # },
    # {
    #     "name": "upper",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/upper",
    #     "split": "train"
    # },
    # {
    #     "name": "sztu",
    #     "root_dir": "/home/data1/szq/Megadepth/benchemarkdata/GAU/sztu",
    #     "split": "train"
    # },
    # ... 你可以继续把其他几十个场景粘贴在这里 ...





    
]

# ================= 工具函数 =================

def find_valid_folder(root_path, candidates):
    """在 root_path 下寻找存在的子文件夹"""
    if not os.path.exists(root_path):
        return None
    for sub in candidates:
        full_path = os.path.join(root_path, sub)
        if os.path.isdir(full_path):
            return full_path
    return None

def delete_file(path):
    """删除文件的封装函数"""
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception as e:
        print(f"{Fore.RED}删除失败: {path} - {e}")
    return False

def clean_dataset(configs):
    print(f"{Fore.CYAN}=== 开始数据集清洗任务 ===")
    if DRY_RUN:
        print(f"{Fore.YELLOW}🚧 当前模式: DRY RUN (预演模式) - 不会删除任何文件")
    else:
        print(f"{Fore.RED}🚨 当前模式: REAL RUN (实战模式) - 将永久删除文件！")
    
    print(f"删除阈值: 空洞/0值比例 > {INVALID_RATIO_THRESHOLD * 100}%\n")

    total_deleted_pairs = 0
    total_scanned_files = 0

    for idx, scene in enumerate(configs):
        name = scene['name']
        root = scene['root_dir']
        
        # 1. 寻找路径
        img_dir = find_valid_folder(root, IMG_CANDIDATES)
        npy_dir = find_valid_folder(root, DEPTH_CANDIDATES)
        
        prefix = f"[{idx+1}/{len(configs)}] {name}"

        if not img_dir or not npy_dir:
            print(f"{Fore.RED}{prefix}: 路径缺失，跳过 (Root: {root})")
            continue

        # 2. 匹配文件 (通过文件名stem匹配)
        img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
        npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        
        # 建立映射: filename_no_ext -> full_path
        img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        npy_map = {os.path.splitext(os.path.basename(f))[0]: f for f in npy_files}
        
        # 只有成对的才处理
        common_stems = sorted(list(set(img_map.keys()) & set(npy_map.keys())))
        
        if not common_stems:
            print(f"{Fore.YELLOW}{prefix}: 没有找到匹配的图像-深度对，跳过")
            continue

        scene_deleted_count = 0
        
        # 3. 遍历检查
        pbar = tqdm(common_stems, desc=f"Scanning {name}", leave=False)
        for stem in pbar:
            npy_path = npy_map[stem]
            img_path = img_map[stem]
            
            try:
                # 加载 NPY
                depth = np.load(npy_path)
                
                # 计算统计量
                total_pixels = depth.size
                valid_pixels = np.count_nonzero(depth > 0) # 假设0和负数都是无效的
                zero_ratio = 1.0 - (valid_pixels / total_pixels)
                
                # 判定条件：0值比例过高 (包含全黑情况)
                if zero_ratio > INVALID_RATIO_THRESHOLD:
                    
                    msg = f"  🗑️ [删除] {stem}: 无效比例 {zero_ratio*100:.1f}% > {INVALID_RATIO_THRESHOLD*100}%"
                    
                    if DRY_RUN:
                        # 预演模式：只打印
                        # tqdm.write(f"{Fore.YELLOW}[预演删除] {stem} (Ratio: {zero_ratio:.2f})")
                        pass # 为了不刷屏，这里不打印太多细节，最后统计
                    else:
                        # 实战模式：执行删除
                        del_npy = delete_file(npy_path)
                        del_img = delete_file(img_path)
                        
                        if del_npy and del_img:
                            # tqdm.write(f"{Fore.RED}{msg}") 
                            pass
                    
                    scene_deleted_count += 1
                    
                total_scanned_files += 1
                
            except Exception as e:
                tqdm.write(f"{Fore.RED}Error processing {stem}: {e}")

        # 场景小结
        if scene_deleted_count > 0:
            action_str = "拟删除" if DRY_RUN else "已删除"
            color = Fore.YELLOW if DRY_RUN else Fore.RED
            print(f"{color}✅ {prefix}: {action_str} {scene_deleted_count} 对文件 (总数: {len(common_stems)})")
        else:
            print(f"{Fore.GREEN}🆗 {prefix}: 数据健康，无需删除")
            
        total_deleted_pairs += scene_deleted_count

    print("="*50)
    print(f"{Fore.CYAN}任务结束 Summary:")
    print(f"扫描场景数: {len(configs)}")
    if DRY_RUN:
        print(f"{Fore.YELLOW}预演结果: 将会删除 {total_deleted_pairs} 对 '废片'。")
        print(f"{Fore.WHITE}请将代码中的 DRY_RUN = False 修改后再次运行以执行删除。")
    else:
        print(f"{Fore.RED}最终结果: 实际删除了 {total_deleted_pairs} 对 '废片'。")

if __name__ == "__main__":
    clean_dataset(SCENE_CONFIGS)