import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from datetime import datetime
import warnings
import traceback

# =================================================================================
# 0. 基础设置 & 全局路径
# =================================================================================
warnings.filterwarnings("ignore") 
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# GT 和 推理结果的根目录 (假设所有场景的子文件夹都在这里面)
# 如果你的 GT/Pred 也在完全不同的地方，请告诉我，我也可以把这个改成在 Config 里单独指定
GT_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data2-down"
# PRED_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data-results-4.5k-out"
PRED_ROOT_BASE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Test-data-results-4.5k-out"
# 输出日志的目录
LOG_DIR = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/Logs-4.5k"

MIN_DEPTH = 1e-3
MAX_DEPTH = 400

# =================================================================================
# 1. 场景配置清单 (在这里填入 CSV 的【绝对路径】)
# =================================================================================

SCENE_CONFIG = {
    # === 场景 1: City ===
    "City": {
        # ✅ 修改点：直接填入 CSV 的完整绝对路径
        "csv_path": "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/city/Global_All_Scenes_Angles.csv",
        
        # 对应的子文件夹名
        "sub_folders": [
            "city_ouputfov64", 
            "city_ouputfov74", 
            "city_ouputfov84"
        ]
    },

    # === 场景 2: Factory ===
    "Factory": {
        # ✅ 修改点：直接填入 CSV 的完整绝对路径
        "csv_path": "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/factory/Global_All_Scenes_Angles.csv",
        
        "sub_folders": [
            "factory-output_fov64", 
            "factory-output_fov74", 
            "factory-output_fov84"
        ]
    },
        "school": {
        # ✅ 修改点：直接填入 CSV 的完整绝对路径
        "csv_path": "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/school/Global_All_Scenes_Angles.csv",
        
        "sub_folders": [
            "school-output_fov64", 
            "school-output_fov74", 
            "school-output_fov84"
        ]
    },
        "park": {
        # ✅ 修改点：直接填入 CSV 的完整绝对路径
        "csv_path": "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/park/Global_All_Scenes_Angles.csv",
        
        "sub_folders": [
            "park-output_fov64", 
            "park-output_fov74", 
            "park-output_fov84"
        ]
    },
        "vnice": {
        # ✅ 修改点：直接填入 CSV 的完整绝对路径
        "csv_path": "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/syn-process/vnicefinal/final_result_with_angles.csv",
        
        "sub_folders": [
            "vnicefinal-output_fov64", 
            "vnicefinal-output_fov74", 
            "vnicefinal-output_fov84"
        ]
    },


    
    # 你可以继续添加更多...
}

TASKS = [
    {"name": "Absolute_Metric", "use_median_scaling": False, "desc": "Absolute Scale"},
    {"name": "Relative_Structure", "use_median_scaling": True, "desc": "Relative Structure"}
]

# =================================================================================
# 2. 逻辑封装
# =================================================================================

class MetadataManager:
    def __init__(self, csv_path):
        self.valid = False
        self.lookup = {}
        try:
            # 这里的 csv_path 现在是绝对路径
            self.df = pd.read_csv(csv_path)
            for idx, row in self.df.iterrows():
                if isinstance(row['Folder_A_Name'], str):
                    hash_key = os.path.splitext(row['Folder_A_Name'])[0]
                    self.lookup[hash_key] = row
            self.valid = True
            print(f"    ✅ CSV Loaded: {len(self.lookup)} records from {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"    ❌ CSV Load Error: {e}")
            print(f"       Path was: {csv_path}")

    def get_info(self, hash_name):
        if not self.valid or hash_name not in self.lookup: return None
        row = self.lookup[hash_name]
        
        # Pitch
        pitch_val = row['pitch']
        anchors = [-90, -75, -60, -45, -30]
        closest_pitch = min(anchors, key=lambda x: abs(x - pitch_val))
        
        # Altitude
        alt_group = "Error"
        try:
            frame_str = os.path.splitext(row['Folder_B_Name'])[0].split('_')[-1]
            frame_num = int(frame_str)
            cycle_pos = (frame_num - 1) % 135
            if 0 <= cycle_pos < 45: alt_group = "Alt_Low"
            elif 45 <= cycle_pos < 90: alt_group = "Alt_Mid"
            elif 90 <= cycle_pos < 135: alt_group = "Alt_High"
        except: pass
            
        return {
            "pitch_group": f"Pitch_{closest_pitch}",
            "alt_group": alt_group,
            "b_name": row['Folder_B_Name']
        }

    def get_fov_group(self, folder_name):
        if "64" in folder_name: return "FOV_64"
        if "74" in folder_name: return "FOV_74"
        if "84" in folder_name: return "FOV_84"
        return "FOV_Other"

# def compute_metrics(gt, pred, mask):
#     if mask.sum() == 0: return None
#     gt, pred = gt[mask], pred[mask]
#     abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()
#     rmse = torch.sqrt(torch.mean((gt - pred) ** 2)).item()
#     thresh = torch.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).float().mean().item()
#     return {"abs_rel": abs_rel, "rmse": rmse, "a1": a1}
def compute_metrics(gt, pred, mask):
    # 1. 第一层过滤：基于深度范围 (GT > 0.001)
    # 如果整张图都没有在这个范围内的GT，那确实没法评，只能跳过图片
    if mask.sum() == 0:
        return None
    
    # 提取有效区域 (Flatten成一维向量)
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    
    # 2. 预测值截断 (推荐保留)
    # 虽然我们要跳过像素，但为了防止所有预测值都是0导致全被跳过，
    # 给预测值一个极小的底数是深度估计的标准操作。
    pred_valid = torch.clamp(pred_valid, min=1e-3)

    # ---------------------------------------------------------
    # 核心修改：定义一个内部函数，专门负责“跳过坏像素”求平均
    # ---------------------------------------------------------
    def safe_mean(error_tensor):
        # torch.isfinite 返回一个 Mask，只有非 inf 且非 nan 的地方是 True
        valid_pixel_mask = torch.isfinite(error_tensor)
        
        # 如果经过清洗，一个有效像素都没剩下了，那这张图只能判无效
        if valid_pixel_mask.sum() == 0:
            return float('nan') 
            
        # 只取有效像素求平均
        return error_tensor[valid_pixel_mask].mean().item()

    # --- 计算 AbsRel ---
    # 先算出一张误差图 (向量)
    abs_rel_map = torch.abs(gt_valid - pred_valid) / gt_valid
    # 剔除 map 里的 inf/nan 像素，求平均
    abs_rel = safe_mean(abs_rel_map)

    # --- 计算 RMSE ---
    rmse_map = (gt_valid - pred_valid) ** 2
    rmse = np.sqrt(safe_mean(rmse_map)) # 先平均再开方

    # --- 计算 RMSE log ---
    # log 可能产生 -inf，必须过滤
    log_diff_map = (torch.log(gt_valid) - torch.log(pred_valid)) ** 2
    rmse_log = np.sqrt(safe_mean(log_diff_map))

    # --- 计算 Accuracy ---
    # a1, a2, a3 只要 pred_valid clamp 过了，通常不会出 inf，但也套用一下安全逻辑
    thresh = torch.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    
    # 这里不需要 isfinite，因为 thresh < 1.25 产生的是 0 或 1，肯定 finite
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()

    # 3. 汇总检查
    # 如果 abs_rel 或 rmse 计算结果是 nan (说明所有像素都被跳过了)，则返回 None
    if np.isnan(abs_rel) or np.isnan(rmse):
        return None

    return {
        "abs_rel": abs_rel, 
        "rmse": rmse, 
        "rmse_log": rmse_log,
        "a1": a1, "a2": a2, "a3": a3
    }

def process_single_scene(scene_name, config):
    print(f"\n{'='*60}")
    print(f"🎬 Processing Scene: {scene_name}")
    print(f"{'='*60}")
    
    # 1. 直接读取配置中的绝对路径
    csv_path = config['csv_path']
    print(f"📍 CSV Path: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Critical: CSV path does not exist! Skipping.")
        return

    meta_mgr = MetadataManager(csv_path)
    if not meta_mgr.valid: return

    full_report_lines = [f"Report for Scene: {scene_name}", f"Date: {datetime.now()}", "="*60]
    debug_logs = []

    # 2. 遍历任务
    for task in TASKS:
        results = []
        task_name = task['name']
        print(f"  🔹 Task: {task_name}...")

        # 3. 遍历子文件夹
        for sub_folder in config['sub_folders']:
            fov_label = meta_mgr.get_fov_group(sub_folder)
            gt_dir = os.path.join(GT_ROOT_BASE, sub_folder, "depth")
            pred_dir = os.path.join(PRED_ROOT_BASE, sub_folder)
            
            if not os.path.exists(pred_dir):
                print(f"    ⚠️ Missing folder: {sub_folder}")
                continue
            
            pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
            
            for pred_path in tqdm(pred_files, desc=f"    {sub_folder}", leave=False):
                try:
                    filename = os.path.basename(pred_path)
                    hash_name = os.path.splitext(filename)[0]
                    info = meta_mgr.get_info(hash_name)
                    
                    # [DEBUG] 收集映射信息
                    if task_name == TASKS[0]['name']:
                        debug_logs.append({
                            "FOV_Folder": sub_folder,
                            "Hash": hash_name,
                            "Found_In_CSV": "YES" if info else "NO",
                            "Frame": info['b_name'] if info else "N/A",
                            "Alt_Group": info['alt_group'] if info else "N/A"
                        })

                    if not info: continue

                    gt_path = os.path.join(gt_dir, filename)
                    if not os.path.exists(gt_path): continue
                    
                    # 读取与计算
                    pred_t = torch.from_numpy(np.load(pred_path)).float().cuda().squeeze()
                    gt_t = torch.from_numpy(np.load(gt_path)).float().cuda().squeeze()
                    
                    if pred_t.shape != gt_t.shape:
                         pred_t = F.interpolate(pred_t.view(1,1,*pred_t.shape), size=gt_t.shape, mode='bilinear').squeeze()

                    mask = (gt_t > MIN_DEPTH) & (gt_t < MAX_DEPTH)
                    # === [添加这行 Debug] ===
                    # 打印总像素数 vs 有效像素数
                    total_pixels = gt_t.numel()
                    valid_pixels = mask.sum().item()
                    print(f"    [DEBUG] {filename}: Total={total_pixels}, Valid={valid_pixels} (Drop Rate: {100*(1-valid_pixels/total_pixels):.1f}%)")


                    if task['use_median_scaling']:
                        valid_gt, valid_pred = gt_t[mask], pred_t[mask]
                        if len(valid_gt) > 0:
                            ratio = torch.median(valid_gt) / torch.median(valid_pred)
                            pred_t *= ratio
                    
                    m = compute_metrics(gt_t, pred_t, mask)
                    if m:
                        m.update({"fov": fov_label, "pitch": info['pitch_group'], "altitude": info['alt_group']})
                        results.append(m)
                        
                except Exception: pass
        
        # 4. 生成报告段落
        if results:
            df = pd.DataFrame(results)
            full_report_lines.append(f"\n>>> TASK: {task_name} <<<")
            full_report_lines.append(f"Overall:\n{df[['abs_rel', 'rmse', 'a1']].mean().to_string()}")
            
            for group_col in ['fov', 'altitude', 'pitch']:
                try:
                    # 排序逻辑
                    if group_col == 'altitude':
                        cats = ["Alt_Low", "Alt_Mid", "Alt_High"]
                        valid = [x for x in cats if x in df[group_col].unique()]
                        df[group_col] = pd.Categorical(df[group_col], categories=valid, ordered=True)
                    elif group_col == 'pitch':
                        cats = ["Pitch_-90", "Pitch_-75", "Pitch_-60", "Pitch_-45", "Pitch_-30"]
                        valid = [x for x in cats if x in df[group_col].unique()]
                        df[group_col] = pd.Categorical(df[group_col], categories=valid, ordered=True)
                    
                    res_str = df.groupby(group_col)[['abs_rel', 'rmse', 'a1']].mean().to_string()
                    full_report_lines.append(f"\nBy {group_col.capitalize()}:\n{res_str}")
                except: pass
            full_report_lines.append("-" * 40)

    # =================================================================================
    # 5. 保存输出
    # =================================================================================
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Debug CSV
    if debug_logs:
        debug_path = os.path.join(LOG_DIR, f"Debug_Check_List_{scene_name}.csv")
        pd.DataFrame(debug_logs).to_csv(debug_path, index=False)
        print(f"    📝 Debug CSV Saved: {debug_path}")

    # 评估报告
    report_path = os.path.join(LOG_DIR, f"Report_Batch_{scene_name}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(full_report_lines))
    print(f"    📄 Report Saved: {report_path}")

# =================================================================================
# 3. 批量入口
# =================================================================================
if __name__ == "__main__":
    print(f"🚀 Starting Batch Evaluation for {len(SCENE_CONFIG)} scenes...")
    
    for scene_name, config in SCENE_CONFIG.items():
        try:
            process_single_scene(scene_name, config)
        except Exception as e:
            print(f"❌ ERROR in {scene_name}: {e}")
            traceback.print_exc()
            
    print("\n🎉 All tasks completed.")