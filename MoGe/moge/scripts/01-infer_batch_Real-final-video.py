# import os
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# import sys
# import json
# import argparse
# import numpy as np
# import cv2
# import torch
# from pathlib import Path
# from tqdm import tqdm

# # ================= ⚙️ 1. 用户配置区域 =================

# PROJECT_ROOT = "/home/szq/moge2/MoGe"
# MODEL_PATH = "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-2/checkpoint/00016500_ema.pt"

# # 视频输入路径
# INPUT_VIDEO = "/home/szq/moge2/MoGe/moge/scripts/org_bf900b90020a1624_1761182830000.mp4" 

# # 结果保存目录
# OUTPUT_ROOT = "/data1/szq/data/Val-Video-Results"

# DEFAULT_CONFIG = {
#     "version": "v2",
#     "device": "cuda",
#     "fp16": True,
#     "resize": 1024,       # 强烈建议设置缩放，防止显存爆炸
#     "resolution_level": 9,
    
#     # --- 🆕 视频生成配置 ---
#     "save_video": True,       # 是否合成 .mp4 视频
#     "save_frames_data": False, # 是否还要保存每一帧的 npy/exr (如果只想要视频，设为 False 可节省空间)
#     "video_mode": "concat",    # 'concat': 原图+深度图并排; 'depth': 仅深度图; 'overlay': 叠加
#     "video_frame_stride": 10    # 1=处理每一帧, 2=每2帧处理一次
# }

# # ================= 🔧 2. 环境设置 =================

# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# try:
#     from moge.model import import_model_class_by_version
#     import utils3d
# except ImportError as e:
#     raise ImportError(f"无法导入 MoGe，请检查路径: {PROJECT_ROOT}")

# # ================= 🧠 3. 推理引擎类 =================

# class MoGeInferenceEngine:
#     def __init__(self, model_path, version="v2", device="cuda", fp16=True):
#         self.device = torch.device(device)
#         self.fp16 = fp16
        
#         print(f"\n📦 [Load] {model_path}")
#         ModelClass = import_model_class_by_version(version)
#         self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
#         if self.fp16: self.model.half()
#         print("✅ Model Ready.")

#     def _normalize_depth_for_display(self, depth, percentile=98):
#         """将 float depth 转换为可视化的 RGB (H, W, 3)"""
#         # 过滤无效值
#         valid_mask = (depth > 0) & np.isfinite(depth)
#         if valid_mask.sum() == 0:
#             return np.zeros((*depth.shape, 3), dtype=np.uint8)

#         # 动态范围截断 (防止极值影响颜色)
#         vmin = np.percentile(depth[valid_mask], 100 - percentile)
#         vmax = np.percentile(depth[valid_mask], percentile)
        
#         # 归一化到 0-255
#         depth_norm = (depth - vmin) / (vmax - vmin + 1e-6)
#         depth_norm = np.clip(depth_norm, 0, 1) * 255
#         depth_uint8 = depth_norm.astype(np.uint8)
        
#         # 伪彩色映射 (INFERNO 看起来很高级，也可以用 JET)
#         depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
#         return depth_color

#     def process_video(self, video_path, output_root, config):
#         video_path = Path(video_path)
#         save_dir = Path(output_root) / video_path.stem
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             print(f"❌ 无法打开视频: {video_path}")
#             return

#         # 获取原视频信息
#         orig_fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         stride = config.get('video_frame_stride', 1)
        
#         # 初始化 VideoWriter
#         writer = None
#         output_video_path = save_dir / f"{video_path.stem}_depth_demo.mp4"
        
#         print(f"▶️ 开始处理: {video_path.name}")
#         print(f"   输出路径: {output_video_path}")
        
#         frame_idx = 0
#         processed_count = 0
        
#         try:
#             pbar = tqdm(total=total_frames, unit="frame")
#             while True:
#                 ret, frame_bgr = cap.read()
#                 if not ret: break
                
#                 if frame_idx % stride == 0:
#                     # 1. 预处理 (Resize)
#                     h, w = frame_bgr.shape[:2]
#                     process_bgr = frame_bgr
#                     resize_to = config.get('resize')
                    
#                     if resize_to is not None:
#                         scale = resize_to / max(h, w)
#                         if scale < 1.0:
#                             new_w = (int(w * scale) // 14) * 14
#                             new_h = (int(h * scale) // 14) * 14
#                             process_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

#                     # 2. 推理
#                     with torch.inference_mode():
#                         img_tensor = torch.from_numpy(cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB))
#                         img_tensor = img_tensor.float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
#                         if self.fp16: img_tensor = img_tensor.half()
                        
#                         output = self.model.infer(img_tensor, resolution_level=config.get('resolution_level', 9))
#                         depth = output['depth'].squeeze().cpu().numpy() # (H, W)

#                     # 3. 生成可视化图
#                     depth_color = self._normalize_depth_for_display(depth)

#                     # 4. 组合视频帧 (Concat 或 Only Depth)
#                     mode = config.get('video_mode', 'concat')
#                     if mode == 'concat':
#                         # 左右拼接: [原图 | 深度图]
#                         final_frame = np.hstack((process_bgr, depth_color))
#                     elif mode == 'depth':
#                         final_frame = depth_color
#                     else:
#                         final_frame = depth_color # 默认

#                     # 5. 初始化 Writer (只在第一帧做，确保尺寸正确)
#                     if writer is None and config.get('save_video', True):
#                         vh, vw = final_frame.shape[:2]
#                         fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Mac下可能需要 'avc1'，Linux通常 'mp4v' 或 'XVID'
#                         # FPS 调整：如果 stride>1，生成的视频会加速。如果想保持原速，fps = orig_fps / stride
#                         save_fps = orig_fps # 保持原帧率 (看起来会加速) 
#                         writer = cv2.VideoWriter(str(output_video_path), fourcc, save_fps, (vw, vh))
                    
#                     # 6. 写入视频
#                     if writer:
#                         writer.write(final_frame)

#                     # 7. (可选) 保存每一帧的 Raw Data
#                     if config.get('save_frames_data'):
#                         frame_dir = save_dir / f"frame_{frame_idx:05d}"
#                         frame_dir.mkdir(exist_ok=True)
#                         np.save(frame_dir / 'depth.npy', depth)
#                         # 如果需要 points，这里也可以保存...

#                     processed_count += 1
                
#                 frame_idx += 1
#                 pbar.update(1)

#         except KeyboardInterrupt:
#             print("🛑 用户中断处理...")
#         finally:
#             pbar.close()
#             cap.release()
#             if writer:
#                 writer.release()
#             print(f"✅ 完成！视频已保存至: {output_video_path}")

# # ================= 🚀 4. 执行 =================

# if __name__ == "__main__":
#     engine = MoGeInferenceEngine(
#         model_path=MODEL_PATH,
#         version=DEFAULT_CONFIG['version'],
#         fp16=DEFAULT_CONFIG['fp16']
#     )
    
#     # 执行视频推理
#     if os.path.isfile(INPUT_VIDEO):
#         engine.process_video(INPUT_VIDEO, OUTPUT_ROOT, DEFAULT_CONFIG)
#     else:
#         print(f"❌ 找不到视频文件: {INPUT_VIDEO}")

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# ================= ⚙️ 1. 用户配置区域 =================

PROJECT_ROOT = "/home/szq/moge2/MoGe"
# MODEL_PATH = "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-2/checkpoint/00016500_ema.pt"

MODEL_PATH = "/home/szq/moge2/MoGe/vitb-normal.pt"



# 视频输入路径
INPUT_VIDEO = "/home/szq/moge2/MoGe/moge/scripts/org_bf900b90020a1624_1761182830000.mp4" 

# 结果保存目录
OUTPUT_ROOT = "/data1/szq/data/Val-Video-Results_MultiViz-origin-b" # 改个输出目录名区分一下

DEFAULT_CONFIG = {
    "version": "v2",
    "device": "cuda",
    "fp16": True,
    "resize": 1024,
    "resolution_level": 9,
    
    # --- 🆕 视频生成配置 ---
    "save_video": True,
    "save_frames_data": False,
    # 视频模式: 'simple'(旧版左右拼接) 或 'advanced'(新版四宫格+标尺+探测点)
    "video_mode": "advanced",  
    "video_frame_stride": 10,
    
    # --- 📍 深度探测点配置 (相对坐标 0.0~1.0: [横向X, 纵向Y]) ---
    "probes_pos": [
        [0.5, 0.5],  # 中心
        [0.25, 0.5], # 左中
        [0.75, 0.5], # 右中
        [0.5, 0.75], # 下中
    ]
}

# ================= 🔧 2. 环境设置 =================

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from moge.model import import_model_class_by_version
except ImportError as e:
    raise ImportError(f"无法导入 MoGe，请检查路径: {PROJECT_ROOT}")

# ================= 🧠 3. 推理引擎类 (升级版) =================

class MoGeInferenceEngine:
    def __init__(self, model_path, version="v2", device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        
        print(f"\n📦 [Load] {model_path}")
        ModelClass = import_model_class_by_version(version)
        self.model = ModelClass.from_pretrained(model_path).to(self.device).eval()
        if self.fp16: self.model.half()
        print("✅ Model Ready.")

    def _normalize_depth_base(self, depth, percentile=98):
        """
        基础归一化：将 float depth 转换为 0-255 的 uint8 灰度图，并返回米制的 vmin, vmax
        """
        valid_mask = (depth > 0) & np.isfinite(depth)
        if valid_mask.sum() == 0:
            return np.zeros_like(depth, dtype=np.uint8), 0.0, 1.0

        # 动态范围截断 (防止极值影响颜色)
        vmin_metric = np.percentile(depth[valid_mask], 100 - percentile)
        vmax_metric = np.percentile(depth[valid_mask], percentile)
        
        # 归一化到 0-255
        depth_norm = (depth - vmin_metric) / (vmax_metric - vmin_metric + 1e-6)
        depth_norm = np.clip(depth_norm, 0, 1) * 255
        depth_uint8 = depth_norm.astype(np.uint8)
        
        return depth_uint8, vmin_metric, vmax_metric

    def _draw_depth_probes(self, img_viz, depth_metric, probes_pos, color=(0, 255, 0)):
        """在图像上绘制深度探测点和数值"""
        h, w = img_viz.shape[:2]
        img_with_probes = img_viz.copy()
        
        for i, (rx, ry) in enumerate(probes_pos):
            cx, cy = int(rx * w), int(ry * h)
            # 确保坐标在范围内
            cx = np.clip(cx, 0, w-1)
            cy = np.clip(cy, 0, h-1)
            
            # 获取真实深度值
            val = depth_metric[cy, cx]
            
            # 绘制标记 (十字)
            cv2.drawMarker(img_with_probes, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            # 绘制文字背景（增加可读性）
            text = f"{val:.2f}m"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_with_probes, (cx + 5, cy - text_h - 5), (cx + 5 + text_w, cy + 5), (0,0,0), -1)
            # 绘制文字
            cv2.putText(img_with_probes, text, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return img_with_probes

    def _draw_colorbar(self, height, width, vmin, vmax, colormap):
        """绘制垂直标尺条"""
        # 创建 0-255 的垂直渐变
        gradient = np.linspace(255, 0, height, dtype=np.uint8)
        gradient = np.tile(gradient[:, np.newaxis], (1, width))
        
        # 上色
        cbar = cv2.applyColorMap(gradient, colormap)
        
        # 添加文字标签 (顶部最大值，底部最小值)
        # 背景框
        cv2.rectangle(cbar, (0, 0), (width, 30), (255,255,255), -1)
        cv2.rectangle(cbar, (0, height-30), (width, height), (255,255,255), -1)
        
        cv2.putText(cbar, f"{vmax:.1f}m", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(cbar, f"{vmin:.1f}m", (5, height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return cbar

    def process_video(self, video_path, output_root, config):
        video_path = Path(video_path)
        save_dir = Path(output_root) / video_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = config.get('video_frame_stride', 1)
        mode = config.get('video_mode', 'advanced')
        
        writer = None
        # 输出文件名加上模式后缀
        output_video_path = save_dir / f"{video_path.stem}_{mode}_viz.mp4"
        
        print(f"▶️ 开始处理: {video_path.name} | 模式: {mode}")
        print(f"   输出路径: {output_video_path}")
        
        frame_idx = 0
        
        try:
            pbar = tqdm(total=total_frames, unit="frame")
            while True:
                ret, frame_bgr = cap.read()
                if not ret: break
                
                if frame_idx % stride == 0:
                    # --- 1. 预处理 (Resize) ---
                    h_orig, w_orig = frame_bgr.shape[:2]
                    process_bgr = frame_bgr
                    resize_to = config.get('resize')
                    if resize_to is not None:
                        scale = resize_to / max(h_orig, w_orig)
                        if scale < 1.0:
                            new_w = (int(w_orig * scale) // 14) * 14
                            new_h = (int(h_orig * scale) // 14) * 14
                            process_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    h_proc, w_proc = process_bgr.shape[:2]

                    # --- 2. 推理 ---
                    with torch.inference_mode():
                        img_tensor = torch.from_numpy(cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB))
                        img_tensor = img_tensor.float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                        if self.fp16: img_tensor = img_tensor.half()
                        output = self.model.infer(img_tensor, resolution_level=config.get('resolution_level', 9))
                        depth_metric = output['depth'].squeeze().cpu().numpy() # (H, W) 真实米制深度

                    # --- 3. 可视化组合 ---
                    
                    if mode == 'simple':
                        # 旧版简单拼接
                        depth_uint8, vmin, vmax = self._normalize_depth_base(depth_metric)
                        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                        final_frame = np.hstack((process_bgr, depth_color))
                        
                    elif mode == 'advanced':
                        # --- A. 准备基础数据 ---
                        depth_uint8, vmin, vmax = self._normalize_depth_base(depth_metric)
                        probes = config.get('probes_pos', [])

                        # --- B. 生成不同配色的深度图 ---
                        # 1. Inferno (主图，带探测点)
                        viz_inferno = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                        viz_inferno_probes = self._draw_depth_probes(viz_inferno, depth_metric, probes, color=(255, 255, 255))
                        # 在角落标记当前帧的动态范围
                        cv2.putText(viz_inferno_probes, f"Range: [{vmin:.1f}m, {vmax:.1f}m]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        # 2. Jet (彩虹色)
                        viz_jet = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                        cv2.putText(viz_jet, "Colormap: JET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        
                        # 3. Viridis (科研常用)
                        viz_viridis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
                        cv2.putText(viz_viridis, "Colormap: VIRIDIS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        # --- C. 处理原图 (带探测点) ---
                        viz_rgb_probes = self._draw_depth_probes(process_bgr, depth_metric, probes, color=(0, 255, 0))
                        cv2.putText(viz_rgb_probes, "Input RGB + Probes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        # --- D. 生成标尺 (Colorbar) ---
                        cbar_width = 60
                        cbar = self._draw_colorbar(h_proc, cbar_width, vmin, vmax, cv2.COLORMAP_INFERNO)

                        # --- E. 组合四宫格布局 ---
                        # 上排: [原图+探测点] | [Inferno深度+探测点] | [标尺]
                        top_row = np.hstack((viz_rgb_probes, viz_inferno_probes, cbar))
                        
                        # 下排: [Jet深度] | [Viridis深度] | [空白填充(为了对齐标尺)]
                        # 创建一个纯黑填充块
                        empty_block = np.zeros((h_proc, cbar_width, 3), dtype=np.uint8)
                        bottom_row = np.hstack((viz_jet, viz_viridis, empty_block))
                        
                        # 最终堆叠: 上排 + 下排
                        final_frame = np.vstack((top_row, bottom_row))

                    # --- 4. 视频写入 ---
                    if writer is None and config.get('save_video', True):
                        vh, vw = final_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(str(output_video_path), fourcc, orig_fps, (vw, vh))
                    
                    if writer:
                        writer.write(final_frame)

                    # (可选) 保存每一帧数据
                    if config.get('save_frames_data'):
                        frame_dir = save_dir / f"frame_{frame_idx:05d}"
                        frame_dir.mkdir(exist_ok=True)
                        np.save(frame_dir / 'depth.npy', depth_metric)

                frame_idx += 1
                pbar.update(1)

        except KeyboardInterrupt:
            print("\n🛑 用户中断处理...")
        finally:
            pbar.close()
            cap.release()
            if writer:
                writer.release()
            print(f"✅ 完成！高级可视化视频已保存至: {output_video_path}")

# ================= 🚀 4. 执行 =================

if __name__ == "__main__":
    # 确保输出目录存在，方便直接运行
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    
    engine = MoGeInferenceEngine(
        model_path=MODEL_PATH,
        version=DEFAULT_CONFIG['version'],
        fp16=DEFAULT_CONFIG['fp16']
    )
    
    if os.path.isfile(INPUT_VIDEO):
        engine.process_video(INPUT_VIDEO, OUTPUT_ROOT, DEFAULT_CONFIG)
    else:
        print(f"❌ 找不到视频文件: {INPUT_VIDEO}")