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
# 1. 全局配置区域
# =================================================================================

# 场景名称
SCENE_NAME = "city"

# 元数据 CSV 路径 (必须包含 Folder_A_Name, Folder_B_Name, pitch)
CSV_PATH = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/city/Global_All_Scenes_Angles.csv"

# GT 根目录 (脚本会自动寻找其下的 city_ouputfovXX/depth)
GT_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data2-down"

# 预测结果根目录 (脚本会自动寻找其下的 city_ouputfovXX)
PRED_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data-results-out"

# 待处理的子文件夹 (FOV分组)
SUB_FOLDERS = [
    "city_ouputfov64",
    "city_ouputfov74",
    "city_ouputfov84"
]

# 输出日志目录
LOG_DIR = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Logs"

# 评估参数
MIN_DEPTH = 1e-3
MAX_DEPTH = 500

# 定义两个评估任务
TASKS = [
    {
        "name": "Absolute_Metric",
        "use_median_scaling": False,  # 绝对评估 (严禁定标)
        "desc": "Evaluating Absolute Scale accuracy (No Scaling)"
    },
    {
        "name": "Relative_Structure",
        "use_median_scaling": True,   # 相对评估 (开启定标)
        "desc": "Evaluating Structural accuracy (With Median Scaling)"
    }
]

# =================================================================================
# 2. 元数据管理器 (Metadata Manager)
# =================================================================================

class MetadataManager:
    def __init__(self, csv_path):
        print(f"📖 Loading metadata from: {csv_path}")
        try:
            self.df = pd.read_csv(csv_path)
            # 建立 Hash -> Row 的映射字典
            self.lookup = {}
            for idx, row in self.df.iterrows():
                if isinstance(row['Folder_A_Name'], str):
                    # 去除后缀 (例如 "abc.jpg" -> "abc")
                    hash_key = os.path.splitext(row['Folder_A_Name'])[0]
                    self.lookup[hash_key] = row
            print(f"✅ Metadata loaded. Total records: {len(self.lookup)}")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            self.lookup = {}

    def get_info(self, hash_name):
        """一次性获取所有元数据，返回字典"""
        if hash_name not in self.lookup:
            return None
        
        row = self.lookup[hash_name]
        b_name = row['Folder_B_Name'] # frame_0001.jpg
        pitch_val = row['pitch']
        
        # 1. 解析 Pitch Group
        anchors = [-90, -75, -60, -45, -30]
        closest_pitch = min(anchors, key=lambda x: abs(x - pitch_val))
        pitch_group = f"Pitch_{closest_pitch}"
        
        # 2. 解析 Altitude Group
        alt_group = "Error"
        try:
            # 提取 frame_xxxx 中的数字
            frame_str = os.path.splitext(b_name)[0].split('_')[-1] # "0001"
            frame_num = int(frame_str)
            
            # 核心逻辑: (N-1) % 135
            cycle_pos = (frame_num - 1) % 135
            
            if 0 <= cycle_pos < 45:
                alt_group = "Alt_Low"
            elif 45 <= cycle_pos < 90:
                alt_group = "Alt_Mid"
            elif 90 <= cycle_pos < 135:
                alt_group = "Alt_High"
        except:
            pass
            
        return {
            "b_name": b_name,
            "pitch_group": pitch_group,
            "alt_group": alt_group,
            "raw_pitch": pitch_val
        }

    def get_fov_group(self, folder_name):
        if "fov64" in folder_name: return "FOV_64"
        if "fov74" in folder_name: return "FOV_74"
        if "fov84" in folder_name: return "FOV_84"
        return "FOV_Other"

# =================================================================================
# 3. 核心评估函数
# =================================================================================

def compute_metrics(gt, pred, mask):
    """计算单个样本的指标"""
    if mask.sum() == 0: return None
    
    gt = gt[mask]
    pred = pred[mask]
    
    # 核心指标
    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()
    sq_rel = torch.mean(((gt - pred) ** 2) / gt).item()
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2)).item()
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2)).item()
    
    # 准确率指标 (delta < 1.25, 1.25^2, 1.25^3)
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()
    
    return {
        "abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, 
        "rmse_log": rmse_log, "a1": a1, "a2": a2, "a3": a3
    }

def process_evaluation():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. 初始化元数据
    meta_mgr = MetadataManager(CSV_PATH)
    if not meta_mgr.lookup: return

    # 为了生成 Debug CSV，我们需要收集一次文件信息 (只收集一次)
    debug_logs = []
    
    # 主报告内容缓冲
    full_report_lines = []
    full_report_lines.append(f"Evaluation Report: {SCENE_NAME}")
    full_report_lines.append(f"Time: {datetime.now()}")
    full_report_lines.append("="*80)

    # 2. 遍历任务 (先跑绝对，再跑相对)
    for task in TASKS:
        task_name = task['name']
        use_ms = task['use_median_scaling']
        
        print(f"\n🚀 Running Task: {task_name} (Median Scaling: {use_ms})")
        results = [] # 存储当前任务的所有结果
        
        # 遍历子文件夹
        for sub_folder in SUB_FOLDERS:
            fov_label = meta_mgr.get_fov_group(sub_folder)
            
            gt_dir = os.path.join(GT_ROOT_BASE, sub_folder, "depth")
            pred_dir = os.path.join(PRED_ROOT_BASE, sub_folder)
            
            if not os.path.exists(pred_dir):
                print(f"⚠️ Folder not found: {pred_dir}")
                continue
                
            pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
            
            # 进度条
            for pred_path in tqdm(pred_files, desc=f"   {sub_folder}", leave=False):
                # 解析文件名
                filename = os.path.basename(pred_path)      # hash.npy
                hash_name = os.path.splitext(filename)[0]   # hash
                
                # 获取元数据
                info = meta_mgr.get_info(hash_name)
                
                # [DEBUG 收集] 仅在第一个任务时收集，避免重复
                if task_name == TASKS[0]['name']:
                    debug_entry = {
                        "FOV_Folder": sub_folder,
                        "Hash_Name": hash_name,
                        "Frame_Name_B": info['b_name'] if info else "Not_Found",
                        "Pitch": info['pitch_group'] if info else "Unknown",
                        "Altitude": info['alt_group'] if info else "Unknown",
                        "Valid": "YES" if info else "NO"
                    }
                    debug_logs.append(debug_entry)
                
                if not info: continue # 没有元数据无法分组，跳过统计
                
                # 加载数据
                try:
                    # GT 路径假设同名
                    gt_path = os.path.join(gt_dir, filename)
                    if not os.path.exists(gt_path): continue
                    
                    pred_np = np.load(pred_path)
                    gt_np = np.load(gt_path)
                    
                    pred_t = torch.from_numpy(pred_np).float().cuda().squeeze()
                    gt_t = torch.from_numpy(gt_np).float().cuda().squeeze()
                    
                    # 尺寸对齐
                    if pred_t.shape != gt_t.shape:
                         pred_t = F.interpolate(pred_t.view(1,1,*pred_t.shape), size=gt_t.shape, mode='bilinear').squeeze()

                    # Mask
                    mask = (gt_t > MIN_DEPTH) & (gt_t < MAX_DEPTH)
                    
                    # 定标 (Median Scaling)
                    if use_ms:
                        valid_gt = gt_t[mask]
                        valid_pred = pred_t[mask]
                        if len(valid_gt) > 0 and len(valid_pred) > 0:
                            ratio = torch.median(valid_gt) / torch.median(valid_pred)
                            pred_t *= ratio
                    
                    # 计算指标
                    m = compute_metrics(gt_t, pred_t, mask)
                    if m:
                        # 记录这一帧的所有信息
                        m.update({
                            "fov": fov_label,
                            "pitch": info['pitch_group'],
                            "altitude": info['alt_group']
                        })
                        results.append(m)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # --- 当前任务统计结束，生成报告数据 ---
        if not results:
            print("❌ No results generated for this task.")
            continue
            
        df = pd.DataFrame(results)
        
        # 写入大报告
        full_report_lines.append(f"\n>>> TASK: {task_name} <<<")
        full_report_lines.append(f"Description: {task['desc']}")
        full_report_lines.append("-" * 40)
        
        # 1. Overall
        full_report_lines.append("1. Overall Performance:")
        full_report_lines.append(df[['abs_rel', 'rmse', 'a1']].mean().to_string())
        
        # 2. By FOV
        full_report_lines.append("\n2. Analysis by FOV:")
        full_report_lines.append(df.groupby('fov')[['abs_rel', 'rmse', 'a1']].mean().to_string())
        
        # 3. By Altitude (指定顺序)
        full_report_lines.append("\n3. Analysis by Altitude:")
        alt_order = ["Alt_Low", "Alt_Mid", "Alt_High"]
        # 确保只列出存在的类别
        valid_alt = [x for x in alt_order if x in df['altitude'].unique()]
        if valid_alt:
            df['altitude'] = pd.Categorical(df['altitude'], categories=valid_alt, ordered=True)
            full_report_lines.append(df.groupby('altitude')[['abs_rel', 'rmse', 'a1']].mean().to_string())
        else:
            full_report_lines.append(df.groupby('altitude')[['abs_rel', 'rmse', 'a1']].mean().to_string())
            
        # 4. By Pitch (指定顺序)
        full_report_lines.append("\n4. Analysis by Pitch:")
        pitch_order = ["Pitch_-90", "Pitch_-75", "Pitch_-60", "Pitch_-45", "Pitch_-30"]
        valid_pitch = [x for x in pitch_order if x in df['pitch'].unique()]
        if valid_pitch:
            df['pitch'] = pd.Categorical(df['pitch'], categories=valid_pitch, ordered=True)
            full_report_lines.append(df.groupby('pitch')[['abs_rel', 'rmse', 'a1']].mean().to_string())
        else:
            full_report_lines.append(df.groupby('pitch')[['abs_rel', 'rmse', 'a1']].mean().to_string())
            
        full_report_lines.append("\n" + "="*80)

    # =================================================================================
    # 4. 保存输出文件
    # =================================================================================
    
    # A. 保存 Debug CSV
    if debug_logs:
        debug_csv_path = os.path.join(LOG_DIR, f"Debug_Check_List_{SCENE_NAME}.csv")
        pd.DataFrame(debug_logs).to_csv(debug_csv_path, index=False)
        print(f"\n✅ Debug CSV saved: {debug_csv_path}")
        print("   (Please open this CSV to verify Altitude and Pitch mappings)")

    # B. 保存 最终评估报告
    report_path = os.path.join(LOG_DIR, f"Report_Detail_{SCENE_NAME}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(full_report_lines))
    print(f"✅ Full Report saved: {report_path}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available, using CPU (slow).")
        torch.device("cpu")
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
    process_evaluation()