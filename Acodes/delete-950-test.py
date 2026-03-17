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

# 你的 AAA 数据集路径列表 (图片文件夹路径)
IMAGE_DIRS = [
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/D1-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/dj3-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/hsd1-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/lower-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-PHD-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/SYS-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/sziit-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town1-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town2-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town3-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/xg5-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yingrenshi1-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yingrenshi2-images",
    # "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yuehai-images"
]

# ================= 功能函数 =================

def delete_file(path):
    """删除文件的封装函数"""
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception as e:
        print(f"{Fore.RED}删除失败: {path} - {e}")
    return False

def clean_dataset_AAA(image_dirs):
    print(f"{Fore.CYAN}=== 开始 AAA 数据集清洗任务 ===")
    if DRY_RUN:
        print(f"{Fore.YELLOW}🚧 当前模式: DRY RUN (预演模式) - 不会删除任何文件")
    else:
        print(f"{Fore.RED}🚨 当前模式: REAL RUN (实战模式) - 将永久删除文件！")
    
    print(f"删除阈值: 空洞/0值比例 > {INVALID_RATIO_THRESHOLD * 100}%\n")

    total_deleted_pairs = 0
    total_scanned_files = 0

    for idx, img_dir in enumerate(image_dirs):
        # 1. 推断场景名称和 NPY 路径
        scene_name = os.path.basename(img_dir).replace("-images", "")
        
        # 核心逻辑：路径替换
        if "-images" in img_dir:
            npy_dir = img_dir.replace("-images", "-npy")
        else:
            # 防御性代码
            npy_dir = img_dir + "_npy" 
        
        prefix = f"[{idx+1}/{len(image_dirs)}] {scene_name}"

        # 2. 检查目录是否存在
        if not os.path.exists(img_dir):
            print(f"{Fore.RED}{prefix}: 图片目录缺失，跳过 -> {img_dir}")
            continue
        if not os.path.exists(npy_dir):
            print(f"{Fore.RED}{prefix}: NPY 目录缺失，跳过 -> {npy_dir}")
            continue

        # 3. 匹配文件 (通过文件名stem匹配)
        img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
        npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        
        # 建立映射: filename_no_ext -> full_path
        # 注意：这里假设文件名唯一，如果有同名不同后缀的图片(a.jpg, a.png)，可能会覆盖
        img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        npy_map = {os.path.splitext(os.path.basename(f))[0]: f for f in npy_files}
        
        # 只有成对的才处理 (交集)
        common_stems = sorted(list(set(img_map.keys()) & set(npy_map.keys())))
        
        if not common_stems:
            print(f"{Fore.YELLOW}{prefix}: 没有找到匹配的图像-深度对，跳过")
            continue

        scene_deleted_count = 0
        
        # 4. 遍历检查
        # 使用 tqdm 显示进度
        for stem in tqdm(common_stems, desc=f"Scanning {scene_name}", leave=False):
            npy_path = npy_map[stem]
            img_path = img_map[stem]
            
            try:
                # 加载 NPY
                depth = np.load(npy_path)
                
                # 计算统计量
                total_pixels = depth.size
                # 统计大于0的像素为有效像素 (0和负数视为无效/空洞)
                valid_pixels = np.count_nonzero(depth > 0) 
                
                # 计算无效比例 (1 - 有效比例)
                zero_ratio = 1.0 - (valid_pixels / total_pixels)
                
                # 判定条件：0值比例过高
                if zero_ratio > INVALID_RATIO_THRESHOLD:
                    
                    if DRY_RUN:
                        # 预演模式：不需要做任何事，只是最后统计
                        pass 
                    else:
                        # 实战模式：执行删除
                        del_npy = delete_file(npy_path)
                        del_img = delete_file(img_path)
                        
                        # 如果删除失败，打印错误
                        if not (del_npy and del_img):
                            print(f"{Fore.RED} 删除失败: {stem}")
                    
                    scene_deleted_count += 1
                    
                total_scanned_files += 1
                
            except Exception as e:
                print(f"{Fore.RED}Error processing {stem}: {e}")

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
    print(f"扫描场景数: {len(image_dirs)}")
    if DRY_RUN:
        print(f"{Fore.YELLOW}预演结果: 将会删除 {total_deleted_pairs} 对 '废片'。")
        print(f"{Fore.WHITE}请将代码中的 DRY_RUN = False 修改后再次运行以执行真正删除。")
    else:
        print(f"{Fore.RED}最终结果: 实际删除了 {total_deleted_pairs} 对 '废片'。")

if __name__ == "__main__":
    clean_dataset_AAA(IMAGE_DIRS)