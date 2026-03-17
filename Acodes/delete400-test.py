import os
import numpy as np
import glob
from tqdm import tqdm
from colorama import Fore, Style, init

# 初始化彩色打印
init(autoreset=True)

# ================= 配置区域 =================

# 阈值设置 (大于此值的像素会被修改)
THRESHOLD = 1000.0
# 替换后的值 (通常设为 0 代表无效/天空)
REPLACE_VALUE = 0

# 你的 AAA 数据集路径列表 (图片文件夹路径)
IMAGE_DIRS = [
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/D1-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/dj3-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/hsd1-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/lower-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-PHD-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/SYS-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/sziit-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town1-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town2-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town3-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/xg5-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yingrenshi1-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yingrenshi2-images",
    "/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yuehai-images"
]

# ================= 功能函数 =================

def batch_clean_values(image_dirs):
    print(f"{Fore.CYAN}🚀 开始批量处理 {len(image_dirs)} 个场景...")
    print(f"{Fore.CYAN}🎯 目标: 将 NPY 中 > {THRESHOLD} 的深度值替换为 {REPLACE_VALUE}")
    print("="*60)

    total_scenes = len(image_dirs)
    
    for idx, img_dir in enumerate(image_dirs):
        # 1. 推断场景名称和 NPY 路径
        scene_name = os.path.basename(img_dir).replace("-images", "")
        
        # 核心逻辑：路径替换
        if "-images" in img_dir:
            npy_dir = img_dir.replace("-images", "-npy")
        else:
            # 防御性代码，防止路径命名不规范
            npy_dir = img_dir + "_npy" 
            print(f"{Fore.YELLOW}⚠️  路径不含 '-images'，猜测 NPY 路径为: {npy_dir}")

        prefix = f"[{idx+1}/{total_scenes}] {scene_name}"

        # 2. 检查文件夹是否存在
        if not os.path.exists(npy_dir):
            print(f"{Fore.RED}❌ {prefix}: NPY 文件夹不存在 -> {npy_dir}")
            continue

        # 3. 获取所有 .npy 文件
        npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
        if not npy_files:
            print(f"{Fore.YELLOW}⚠️  {prefix}: 文件夹存在但为空 -> {npy_dir}")
            continue

        # 4. 循环处理文件
        modified_count = 0
        
        # 使用 tqdm 显示当前场景进度
        for f_path in tqdm(npy_files, desc=f"Processing {scene_name}", leave=False):
            try:
                # 加载
                data = np.load(f_path)
                
                # 检查是否需要修改 (先用 mask 判断，避免无意义的 IO 操作)
                mask = data > THRESHOLD
                
                if np.any(mask):
                    # 修改数值
                    data[mask] = REPLACE_VALUE
                    # 覆盖保存
                    np.save(f_path, data)
                    modified_count += 1
                    
            except Exception as e:
                print(f"{Fore.RED}❌ {prefix}: 读写文件失败 {os.path.basename(f_path)} - {e}")

        # 场景小结
        if modified_count > 0:
            print(f"{Fore.GREEN}✅ {prefix}: 已清理 {modified_count}/{len(npy_files)} 个文件 (Dir: {os.path.basename(npy_dir)})")
        else:
            print(f"{Fore.WHITE}🆗 {prefix}: 无需修改 (所有值均 <= {THRESHOLD})")

    print("="*60)
    print(f"{Fore.GREEN}🎉 所有场景处理完毕！")

# ================= 执行入口 =================

if __name__ == "__main__":
    if not IMAGE_DIRS:
        print(f"{Fore.RED}❌ 错误：IMAGE_DIRS 列表为空！")
    else:
        print(f"{Fore.YELLOW}!!! 警告：此操作将永久修改原始 .npy 文件 (Values > {THRESHOLD} -> {REPLACE_VALUE}) !!!")
        user_input = input("确认执行吗? (输入 'yes' 继续): ")
        
        if user_input.lower() == 'yes':
            batch_clean_values(IMAGE_DIRS)
        else:
            print("操作已取消。")