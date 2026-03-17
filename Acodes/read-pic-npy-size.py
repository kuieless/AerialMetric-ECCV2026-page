import os
import cv2
import numpy as np
import glob
from colorama import Fore, Style, init

# 初始化彩色打印
init(autoreset=True)

# ================= 配置区域 =================

# BASE_DIR = "/home/data1/szq/Megadepth/benchemarkdata/ODM"
BASE_DIR = "/home/data1/szq/Megadepth/benchemarkdata/ODM"

# 将你的场景名直接粘贴在这个字符串里（空格换行都没关系，脚本会自动处理）
'''
# aukerman  brighton-beach    garfield_msp                lewis  ODM2    ODM3    ODM-34-1  ODM37  ODM6    ODM8-2  park10  park12  park14  park3  park5  park7  park9
# bellus    caliterra         ODM1   ODM-27  ODM-32  ODM-34-2  ODM39  ODM8-1  park0   park11  park13  park2   park4  park6  park8  seneca
'''
# SCENE_STRING = """
# brighton-beach-npy          dj3-images   lower-images  park13-images  R1-PHD-images  SYS-images    tasks_output.txt  town2-npy     xg5-npy             yingrenshi2-npy
# bellus-images                   D1-images                   dj3-npy      lower-npy     park13-npy     R1-PHD-npy     SYS-npy       town1-images      town3-images  yingrenshi1-images  yuehai-images
# bellus-npy                      D1-npy                      hsd1-images  ODM1-images   R1-images      seneca-images  sziit-images  town1-npy         town3-npy     yingrenshi1-npy     yuehai-npy
# brighton-beach-images           dataset_quality_report.txt  hsd1-npy     ODM1-npy      R1-npy         seneca-npy     sziit-npy     town2-images      xg5-images    yingrenshi2-images
# yuehai10-1  yuehai11-1  yuehai13-1  yuehai14-1  yuehai15-2  yuehai16-2  yuehai2-1  yuehai5-1  yuehai6-1  yuehai7-1  yuehai8-1  yuehai9-1
# yuehai10-2  yuehai12-1  yuehai13-2  yuehai15-1  yuehai16-1  yuehai17-1  yuehai4-1  yuehai5-2  yuehai6-2  yuehai7-2  yuehai8-2  yuehai9-2

# """
SCENE_STRING = """
      
 brighton-beach-npy          dj3-images   lower-images  park13-images  R1-PHD-images  SYS-images    tasks_output.txt  town2-npy     xg5-npy             yingrenshi2-npy
bellus-images                   D1-images                   dj3-npy      lower-npy     park13-npy     R1-PHD-npy     SYS-npy       town1-images      town3-images  yingrenshi1-images  yuehai-images
bellus-npy                      D1-npy                      hsd1-images  ODM1-images   R1-images      seneca-images  sziit-images  town1-npy         town3-npy     yingrenshi1-npy     yuehai-npy
brighton-beach-images           hsd1-npy     ODM1-npy      R1-npy         seneca-npy     sziit-npy     town2-images      xg5-images    yingrenshi2-images

"""

# 文件夹可能的名称优先级
IMG_CANDIDATES = ["images", "rgb", "img", "images_downsampled"]
DEPTH_CANDIDATES = ["npy", "depth", "depth_npy", "npy_downsampled"]

# ===========================================

def find_valid_folder(root_path, candidates):
    """在 root_path 下寻找存在的子文件夹"""
    if not os.path.exists(root_path):
        return None
    for sub in candidates:
        full_path = os.path.join(root_path, sub)
        if os.path.isdir(full_path):
            return full_path
    return None

def get_first_file_shape(folder_path, is_npy=False):
    """读取文件夹中第一个文件的形状"""
    # 查找文件模式
    pattern = "*.npy" if is_npy else "*.*"
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    
    # 过滤掉非图片文件 (如果是图片模式)
    if not is_npy:
        valid_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]

    if not files:
        return None, "No Files"

    first_file = files[0]
    
    try:
        if is_npy:
            data = np.load(first_file)
            # 处理 (H, W, 1) 的情况
            if data.ndim == 3:
                data = data.squeeze()
            return data.shape, os.path.basename(first_file)
        else:
            img = cv2.imread(first_file)
            if img is None:
                return None, "Read Error"
            # cv2 shape is (H, W, C) -> return (H, W)
            return img.shape[:2], os.path.basename(first_file)
    except Exception as e:
        return None, str(e)

def check_odm_scenes():
    # 1. 解析场景列表
    scenes = SCENE_STRING.split()
    scenes = sorted(list(set(scenes))) # 去重并排序

    print(f"{Fore.CYAN}=== ODM 场景分辨率检查 ===")
    print(f"Base Dir: {BASE_DIR}")
    print(f"Total Scenes: {len(scenes)}")
    print("-" * 80)
    
    # 打印表头
    header = f"{'Scene Name':<20} | {'Img Size (HxW)':<15} | {'Npy Size (HxW)':<15} | {'Status':<10}"
    print(header)
    print("-" * 80)

    for scene in scenes:
        scene_path = os.path.join(BASE_DIR, scene)
        
        # 寻找子文件夹
        img_dir = find_valid_folder(scene_path, IMG_CANDIDATES)
        npy_dir = find_valid_folder(scene_path, DEPTH_CANDIDATES)

        img_shape_str = "Missing Dir"
        npy_shape_str = "Missing Dir"
        status = f"{Fore.YELLOW}MISSING"
        
        real_img_shape = None
        real_npy_shape = None

        # 获取图片尺寸
        if img_dir:
            shape, fname = get_first_file_shape(img_dir, is_npy=False)
            if shape:
                real_img_shape = shape
                img_shape_str = f"{shape[0]}x{shape[1]}"
            else:
                img_shape_str = "Empty"

        # 获取NPY尺寸
        if npy_dir:
            shape, fname = get_first_file_shape(npy_dir, is_npy=True)
            if shape:
                real_npy_shape = shape
                npy_shape_str = f"{shape[0]}x{shape[1]}"
            else:
                npy_shape_str = "Empty"

        # 判定状态
        if real_img_shape and real_npy_shape:
            if real_img_shape == real_npy_shape:
                status = f"{Fore.GREEN}MATCH"
            else:
                status = f"{Fore.RED}MISMATCH"
        elif img_shape_str == "Missing Dir" and npy_shape_str == "Missing Dir":
             status = f"{Fore.RED}NOT FOUND"

        # 打印行
        print(f"{scene:<20} | {img_shape_str:<15} | {npy_shape_str:<15} | {status}")

    print("-" * 80)

if __name__ == "__main__":
    check_odm_scenes()