# import os
# import argparse
# import numpy as np
# import cv2
# import torch
# from pathlib import Path
# from tqdm import tqdm

# # 🔥 核心参数：如果是毫米单位，就是 1000；如果是 1/256 米，就是 256
# DEPTH_SCALE = 1000.0 

# def compute_metrics(gt, pred):
#     """计算误差指标"""
#     # 避免除以0
#     gt = np.clip(gt, 1e-3, None)
#     pred = np.clip(pred, 1e-3, None)

#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()

#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())

#     abs_rel = np.mean(np.abs(gt - pred) / gt)
    
#     return np.array([abs_rel, rmse, a1, a2, a3])

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pred_root", type=str, required=True, help="推理结果目录")
#     args = parser.parse_args()
    
#     root = Path(args.pred_root)
#     # 寻找所有的 depth.npy
#     pred_files = list(root.rglob("depth.npy"))
#     print(f"🔍 正在评估 {len(pred_files)} 个结果...")
    
#     metrics_list = []
    
#     for pred_path in tqdm(pred_files):
#         # 假设 GT 就在同一个文件夹里 (depth_gt.png)
#         gt_path = pred_path.parent / "depth_gt.png"
        
#         if not gt_path.exists():
#             continue
            
#         # 1. 读取预测 (NPY)
#         pred = np.load(pred_path)
        
#         # 2. 读取真值 (PNG) - 注意 -1 参数
#         gt_raw = cv2.imread(str(gt_path), -1)
#         if gt_raw is None: continue
        
#         # 转 float 并除以单位
#         gt = gt_raw.astype(np.float32) / DEPTH_SCALE
        
#         # 3. 对齐尺寸
#         h, w = gt.shape
#         if pred.shape != (h, w):
#             pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            
#         # 4. Mask (去掉无效值，通常 0 是无效，太远也是无效)
#         mask = (gt > 0.1) & (gt < 150) # 只评估 0.1米 到 150米
#         if mask.sum() < 10: continue
        
#         # 5. 中值对齐 (Scale Alignment - 可选)
#         # 如果你的模型是 Moge (Metric3D)，通常不需要这一步
#         # 但如果是相对深度模型，必须把下面这行解开注释：
#         # scale = np.median(gt[mask]) / (np.median(pred[mask]) + 1e-8)
#         # pred = pred * scale

#         metrics_list.append(compute_metrics(gt[mask], pred[mask]))
        
#     if not metrics_list:
#         print("❌ 没有有效数据！请检查文件夹里是否有 depth_gt.png")
#         return
        
#     avg = np.mean(metrics_list, axis=0)
    
#     print("\n" + "="*60)
#     print(f"📊 最终评估报告 (共 {len(metrics_list)} 张图片)")
#     print("="*60)
#     print(f"{'AbsRel':>10} | {'RMSE':>10} | {'δ < 1.25':>10} | {'δ < 1.25²':>10} | {'δ < 1.25³':>10}")
#     print("-" * 60)
#     print(f"{avg[0]:10.4f} | {avg[1]:10.4f} | {avg[2]:10.4f} | {avg[3]:10.4f} | {avg[4]:10.4f}")
#     print("="*60)

# if __name__ == "__main__":
#     main()

import os
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# 根据之前的检测，这里是 1000
DEPTH_SCALE = 1000.0 

def compute_metrics(gt, pred):
    # 防止除以0
    gt = np.clip(gt, 1e-3, None)
    pred = np.clip(pred, 1e-3, None)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    
    return np.array([abs_rel, rmse, a1, a2, a3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", type=str, required=True, help="推理结果目录")
    args = parser.parse_args()
    
    root = Path(args.pred_root)
    # 寻找藏在深处的 depth.npy
    pred_files = list(root.rglob("depth.npy"))
    print(f"🔍 正在评估 {len(pred_files)} 个结果...")
    
    metrics_list = []
    
    for pred_path in tqdm(pred_files):
        # 🔥【关键修改】智能查找 GT
        # 因为推理结果可能在 Scene/images/image_name/depth.npy
        # 而 GT 在 Scene/depth_gt.png
        # 所以我们要往上找几层
        
        gt_path = None
        # 检查父目录、爷爷目录、太爷爷目录...
        for parent in [pred_path.parent] + list(pred_path.parents)[:3]:
            if (parent / "depth_gt.png").exists():
                gt_path = parent / "depth_gt.png"
                break
        
        if gt_path is None:
            # print(f"⚠️ 找不到 GT: {pred_path}") # 调试用
            continue
            
        # 读取数据
        pred = np.load(pred_path)
        gt_raw = cv2.imread(str(gt_path), -1)
        if gt_raw is None: continue
        
        # 转 float
        gt = gt_raw.astype(np.float32) / DEPTH_SCALE
        
        # 对齐尺寸
        h, w = gt.shape
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # 评估 Mask (0.1m - 150m)
        mask = (gt > 0.1) & (gt < 150)
        if mask.sum() < 10: continue
        
        metrics_list.append(compute_metrics(gt[mask], pred[mask]))
        
    if not metrics_list:
        print("❌ 没有有效数据！")
        return
        
    avg = np.mean(metrics_list, axis=0)
    
    print("\n" + "="*60)
    print(f"📊 最终评估报告 (共 {len(metrics_list)} 张)")
    print("="*60)
    print(f"{'AbsRel':>10} | {'RMSE':>10} | {'δ < 1.25':>10}")
    print("-" * 60)
    print(f"{avg[0]:10.4f} | {avg[1]:10.4f} | {avg[2]:10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()