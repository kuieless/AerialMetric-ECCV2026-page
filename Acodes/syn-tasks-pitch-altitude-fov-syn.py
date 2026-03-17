import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from datetime import datetime

# =================================================================================
# 1. 配置区域 (请根据实际情况修改)
# =================================================================================

# 场景名称 (用于日志显示)
SCENE_NAME = "city"

# CSV 路径 (包含 pitch, folder_A, folder_B 映射)
CSV_PATH = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/city/Global_All_Scenes_Angles.csv"

# GT 根目录 (包含 fov64/depth, fov74/depth...)
# 注意：代码会自动寻找下面的 city_ouputfovXX/depth
GT_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data2-down"

# 推理结果根目录 (包含 fov64/*.npy)
PRED_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data-results-out"

# 需要评估的子文件夹列表 (对应不同的 FOV)
# 注意：保留你提供的拼写 'ouput'
SUB_FOLDERS = [
    "city_ouputfov64",
    "city_ouputfov74",
    "city_ouputfov84"
]

# 输出报告路径
OUTPUT_REPORT = f"/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Logs/Report_Detail_{SCENE_NAME}.txt"

# 评估设置
MIN_DEPTH = 1e-3
MAX_DEPTH = 400
USE_MEDIAN_SCALING = True # 绝对尺度评估设为 False

# =================================================================================
# 2. 元数据解析管理器 (核心逻辑)
# =================================================================================

class MetadataManager:
    def __init__(self, csv_path):
        print(f"Loading metadata from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # 创建 Hash (Folder_A) -> Row Data 的快速查找字典
        # key: 去除后缀的文件名 (例如 "58d1d8..."), value: row series
        self.lookup = {}
        for idx, row in self.df.iterrows():
            # Folder_A_Name 类似于 "hash.jpg"，我们只存 "hash"
            hash_name = os.path.splitext(row['Folder_A_Name'])[0]
            self.lookup[hash_name] = row

    def get_pitch_group(self, hash_name):
        """根据 Hash 获取 Pitch 分组 (-90, -75, -60, -45, -30)"""
        if hash_name not in self.lookup:
            return "Unknown"
        
        pitch_val = self.lookup[hash_name]['pitch']
        
        # 定义标准锚点
        anchors = [-90, -75, -60, -45, -30]
        # 找到最近的锚点
        closest = min(anchors, key=lambda x: abs(x - pitch_val))
        return f"Pitch_{closest}"

    def get_altitude_group(self, hash_name):
        """
        根据 Frame Number 判断高度组
        规则: 1-45(Low), 46-90(Mid), 91-135(High), 循环 405 帧
        """
        if hash_name not in self.lookup:
            return "Unknown"
        
        # 获取 Folder_B_Name (例如 "frame_0001.jpg")
        frame_name = self.lookup[hash_name]['Folder_B_Name']
        try:
            # 提取数字 "0001" -> 1
            frame_num = int(os.path.splitext(frame_name)[0].split('_')[1])
            
            # 核心数学逻辑: (N-1) % 135
            # 0-44 -> Low, 45-89 -> Mid, 90-134 -> High
            cycle_pos = (frame_num - 1) % 135
            
            if 0 <= cycle_pos < 45:
                return "Alt_Low"
            elif 45 <= cycle_pos < 90:
                return "Alt_Mid"
            elif 90 <= cycle_pos < 135:
                return "Alt_High"
            else:
                return "Error"
        except Exception as e:
            print(f"Error parsing frame number from {frame_name}: {e}")
            return "Error"

    def get_fov_from_folder(self, folder_name):
        """从文件夹名解析 FOV"""
        if "fov64" in folder_name: return "FOV_64"
        if "fov74" in folder_name: return "FOV_74"
        if "fov84" in folder_name: return "FOV_84"
        return "FOV_Other"

# =================================================================================
# 3. 数据加载与评估逻辑
# =================================================================================

def compute_metrics(gt, pred, mask):
    """计算标准深度估计指标"""
    if mask.sum() == 0:
        return None
    
    gt_c = gt[mask]
    pred_c = pred[mask]
    
    # AbsRel
    abs_rel = torch.mean(torch.abs(gt_c - pred_c) / gt_c).item()
    # RMSE
    rmse = torch.sqrt(torch.mean((gt_c - pred_c) ** 2)).item()
    # A1 (delta < 1.25)
    thresh = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    a1 = (thresh < 1.25).float().mean().item()
    
    return {"abs_rel": abs_rel, "rmse": rmse, "a1": a1}

def process_scene():
    # 1. 初始化元数据管理器
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV not found: {CSV_PATH}")
        return
    meta_mgr = MetadataManager(CSV_PATH)
    
    results = []
    
    print(f"🚀 Starting evaluation for scene: {SCENE_NAME}")
    print(f"📂 GT Root: {GT_ROOT_BASE}")
    print(f"📂 Pred Root: {PRED_ROOT_BASE}")
    
    # 2. 遍历三个 FOV 文件夹
    for sub_folder in SUB_FOLDERS:
        fov_label = meta_mgr.get_fov_from_folder(sub_folder)
        
        # 构建具体路径
        # GT: .../city_ouputfov64/depth
        gt_dir = os.path.join(GT_ROOT_BASE, sub_folder, "depth")
        # Pred: .../city_ouputfov64/ (npy直接在里面)
        pred_dir = os.path.join(PRED_ROOT_BASE, sub_folder)
        
        if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
            print(f"⚠️ Skipping {sub_folder}, path missing.")
            continue
            
        # 获取该文件夹下所有 Pred npy 文件
        pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        
        print(f"   Processing {sub_folder} ({len(pred_files)} files)...")
        
        for pred_path in tqdm(pred_files, leave=False):
            # 获取 Hash 文件名 (不带后缀)
            filename = os.path.basename(pred_path) # "hash.npy"
            hash_name = os.path.splitext(filename)[0] # "hash"
            
            # 寻找对应的 GT 文件
            gt_path = os.path.join(gt_dir, filename) # 假设 GT 和 Pred 同名
            
            if not os.path.exists(gt_path):
                # 尝试容错: 有时候 GT 可能是 .png 或其他格式? 
                # 这里假设你的 GT 也是 .npy。如果不是，需要修改这里。
                continue
                
            # --- 读取数据 ---
            try:
                pred_map = torch.from_numpy(np.load(pred_path)).float().cuda().squeeze()
                gt_map = torch.from_numpy(np.load(gt_path)).float().cuda().squeeze()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            # --- 尺寸对齐 ---
            if pred_map.shape != gt_map.shape:
                 pred_map = F.interpolate(pred_map.unsqueeze(0).unsqueeze(0), size=gt_map.shape, mode='bilinear', align_corners=False).squeeze()

            # --- 掩码 ---
            mask = (gt_map > MIN_DEPTH) & (gt_map < MAX_DEPTH)
            
            # --- 定标 (可选) ---
            if USE_MEDIAN_SCALING:
                ratio = torch.median(gt_map[mask]) / torch.median(pred_map[mask])
                pred_map *= ratio
            
            # --- 计算指标 ---
            metrics = compute_metrics(gt_map, pred_map, mask)
            if metrics is None: continue
            
            # --- 获取分类标签 (Pitch, Alt) ---
            pitch_label = meta_mgr.get_pitch_group(hash_name)
            alt_label = meta_mgr.get_altitude_group(hash_name)
            
            # --- 记录 ---
            record = {
                "hash": hash_name,
                "fov": fov_label,
                "pitch": pitch_label,
                "altitude": alt_label,
                "abs_rel": metrics['abs_rel'],
                "rmse": metrics['rmse'],
                "a1": metrics['a1']
            }
            results.append(record)

    # =================================================================================
    # 4. 生成报告
    # =================================================================================
    if not results:
        print("❌ No valid results found.")
        return

    df = pd.DataFrame(results)
    
    # 过滤掉无法匹配 Metadata 的数据
    df = df[df['pitch'] != "Unknown"]
    df = df[df['altitude'] != "Error"]

    lines = []
    lines.append(f"Evaluation Report for Scene: {SCENE_NAME}")
    lines.append(f"Date: {datetime.now()}")
    lines.append(f"Total Images Evaluated: {len(df)}")
    lines.append("="*80)

    # 1. 总体平均
    lines.append("\n>>> Overall Performance <<<")
    overall = df[['abs_rel', 'rmse', 'a1']].mean()
    lines.append(overall.to_string())

    # 2. 按 FOV 分析
    lines.append("\n>>> Analysis by FOV (Field of View) <<<")
    lines.append(df.groupby('fov')[['abs_rel', 'rmse', 'a1']].mean().to_string())

    # 3. 按 Pitch 分析
    lines.append("\n>>> Analysis by Pitch Angle (Camera Tilt) <<<")
    # 自定义排序: -90 到 -30
    pitch_order = ["Pitch_-90", "Pitch_-75", "Pitch_-60", "Pitch_-45", "Pitch_-30"]
    # 仅保留存在的组并排序
    valid_pitch_order = [p for p in pitch_order if p in df['pitch'].unique()]
    if valid_pitch_order:
        df['pitch'] = pd.Categorical(df['pitch'], categories=valid_pitch_order, ordered=True)
        lines.append(df.groupby('pitch')[['abs_rel', 'rmse', 'a1']].mean().to_string())
    else:
        lines.append(df.groupby('pitch')[['abs_rel', 'rmse', 'a1']].mean().to_string())

    # 4. 按 Altitude 分析
    lines.append("\n>>> Analysis by Relative Altitude <<<")
    alt_order = ["Alt_Low", "Alt_Mid", "Alt_High"]
    df['altitude'] = pd.Categorical(df['altitude'], categories=alt_order, ordered=True)
    lines.append(df.groupby('altitude')[['abs_rel', 'rmse', 'a1']].mean().to_string())

    # 5. 交叉分析 (High Altitude 下不同 FOV 的表现)
    lines.append("\n>>> Cross Analysis: FOV performance at High Altitude <<<")
    high_alt_df = df[df['altitude'] == 'Alt_High']
    if not high_alt_df.empty:
        lines.append(high_alt_df.groupby('fov')[['abs_rel', 'rmse', 'a1']].mean().to_string())
    else:
        lines.append("No High Altitude data found.")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Report generated: {OUTPUT_REPORT}")
    print(f"\nSample of DataFrame:\n{df.head()}")

if __name__ == "__main__":
    process_scene()