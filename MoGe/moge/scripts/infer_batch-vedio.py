# # # import os
# # # import subprocess
# # # import shutil
# # # import glob
# # # import cv2
# # # from tqdm import tqdm

# # # # ================= 配置区域 =================

# # # # 1. 输入视频路径
# # # VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1.mp4"

# # # # 2. 输出视频路径
# # # OUTPUT_VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1_depth.mp4"

# # # # 3. 推理脚本和模型路径
# # # SCRIPT_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/infer.py"
# # # MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"

# # # # 4. 临时文件夹
# # # TEMP_DIR = "./temp_video_inference"

# # # # 5. 推理时的显存限制 (可选)
# # # # 如果显存不够，这里可以填 1024 或 800。如果显存够大，设为 None 表示用原图推理
# # # INFERENCE_RESIZE = None  

# # # # ==========================================

# # # def get_video_info(video_path):
# # #     """获取原视频的宽、高、帧率"""
# # #     cap = cv2.VideoCapture(video_path)
# # #     if not cap.isOpened():
# # #         raise ValueError(f"无法打开视频: {video_path}")
    
# # #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #     fps = cap.get(cv2.CAP_PROP_FPS)
# # #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
# # #     cap.release()
# # #     return width, height, fps, total_frames

# # # def run_video_inference():
# # #     # 0. 获取原视频信息
# # #     if not os.path.exists(VIDEO_PATH):
# # #         print(f"❌ 错误: 找不到输入视频: {VIDEO_PATH}")
# # #         return

# # #     print(f"🔍 分析原视频信息...")
# # #     orig_w, orig_h, orig_fps, total_frames = get_video_info(VIDEO_PATH)
# # #     print(f"   - 分辨率: {orig_w} x {orig_h}")
# # #     print(f"   - 帧率: {orig_fps:.2f}")
# # #     print(f"   - 总帧数: {total_frames}")

# # #     # 1. 准备环境
# # #     if os.path.exists(TEMP_DIR):
# # #         shutil.rmtree(TEMP_DIR)
    
# # #     frames_dir = os.path.join(TEMP_DIR, "frames")
# # #     output_frames_dir = os.path.join(TEMP_DIR, "output")
# # #     os.makedirs(frames_dir, exist_ok=True)
# # #     os.makedirs(output_frames_dir, exist_ok=True)

# # #     # 2. 拆帧
# # #     print(f"🎬 [1/3] 正在拆帧...")
# # #     cmd_extract = [
# # #         "ffmpeg", 
# # #         "-i", VIDEO_PATH, 
# # #         "-q:v", "2", 
# # #         os.path.join(frames_dir, "%05d.jpg")
# # #     ]
# # #     subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# # #     # 3. 推理
# # #     print(f"🧠 [2/3] 开始深度估计推理...")
# # #     cmd_inference = [
# # #         "python", SCRIPT_PATH,
# # #         "--input", frames_dir,
# # #         "--output", output_frames_dir,
# # #         "--pretrained", MODEL_PATH,
# # #         "--maps",
# # #         "--fp16"
# # #     ]
    
# # #     # 如果指定了推理时的 Resize (为了省显存)，加进去
# # #     # 注意：这里 Resize 只是为了推理不爆显存，最后生成视频时我们会还原回原尺寸
# # #     if INFERENCE_RESIZE is not None:
# # #         cmd_inference.extend(["--resize", str(INFERENCE_RESIZE)])
    
# # #     env = os.environ.copy()
# # #     env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    
# # #     try:
# # #         subprocess.run(cmd_inference, env=env, check=True)
# # #     except subprocess.CalledProcessError as e:
# # #         print(f"❌ 推理失败: {e}")
# # #         return

# # #     # 4. 合成视频 (关键修改：强制 Resize 回原尺寸)
# # #     print(f"🎞️ [3/3] 正在合成视频 (强制还原尺寸: {orig_w}x{orig_h})...")
    
# # #     depth_frames = sorted(glob.glob(os.path.join(output_frames_dir, "*_depth_vis.png")))
# # #     if not depth_frames:
# # #         print("❌ 未找到生成的深度图。")
# # #         return

# # #     # 初始化视频写入器
# # #     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# # #     video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, orig_fps, (orig_w, orig_h))

# # #     for frame_path in tqdm(depth_frames, desc="Writing Video"):
# # #         img = cv2.imread(frame_path)
        
# # #         # 🔥 核心修改：如果推理出来的图尺寸不对，强行缩放回原视频尺寸
# # #         if img.shape[1] != orig_w or img.shape[0] != orig_h:
# # #             img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
# # #         video.write(img)

# # #     video.release()
# # #     print(f"\n🎉 完美！视频已保存至: {OUTPUT_VIDEO_PATH}")
# # #     print(f"   尺寸: {orig_w}x{orig_h}, FPS: {orig_fps:.2f}")

# # #     # 清理
# # #     shutil.rmtree(TEMP_DIR)

# # # if __name__ == "__main__":
# # #     run_video_inference()

# # import os
# # import subprocess
# # import shutil
# # import glob
# # import cv2
# # import numpy as np
# # from tqdm import tqdm

# # # ================= 配置区域 =================

# # # 1. 输入视频路径
# # VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1.mp4"

# # # 2. 输出视频路径
# # OUTPUT_VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1_depth_sky.mp4"

# # # 3. 脚本与模型路径
# # SCRIPT_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/infer.py"
# # MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"

# # # 4. 临时文件夹
# # TEMP_DIR = "./temp_video_inference_sky"

# # # 5. 天空处理模式
# # # "ORIGINAL": 显示原图原本的天空
# # # "RED_MASK": 原图天空 + 半透明红色遮罩 (方便看出哪里被判定为天空)
# # # "BLUE_MASK": 原图天空 + 半透明蓝色遮罩
# # SKY_MODE = "RED_MASK" 

# # # ==========================================

# # def get_video_info(video_path):
# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened(): raise ValueError(f"无法打开: {video_path}")
# #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     cap.release()
# #     return w, h, fps, frames

# # def process_frame_fusion(depth_vis_path, original_img_path, target_size):
# #     """
# #     核心函数：融合深度图和原图
# #     """
# #     # 1. 读取深度图和原图
# #     depth_vis = cv2.imread(depth_vis_path) # [H, W, 3] BGR
# #     original = cv2.imread(original_img_path)
    
# #     if depth_vis is None or original is None: return None

# #     # 2. 强制调整尺寸对齐
# #     w, h = target_size
# #     if depth_vis.shape[:2] != (h, w):
# #         depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST) # Nearest防止产生伪边缘颜色
# #     if original.shape[:2] != (h, w):
# #         original = cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR)

# #     # 3. 提取天空掩码 (Mask)
# #     # 假设深度图中完全黑色的区域 (0,0,0) 是无效/天空区域
# #     # 转化为灰度更方便判断
# #     gray = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2GRAY)
# #     # 阈值设为 5 而不是 0，防止有些轻微的噪点
# #     _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
# #     # mask: 255是前景(有深度), 0是背景(天空)
    
# #     mask_inv = cv2.bitwise_not(mask) # 255是天空

# #     # 4. 根据模式处理天空
# #     final_frame = depth_vis.copy()
    
# #     if SKY_MODE == "ORIGINAL":
# #         # 直接把原图的天空抠出来，贴进去
# #         sky_region = cv2.bitwise_and(original, original, mask=mask_inv)
# #         foreground = cv2.bitwise_and(depth_vis, depth_vis, mask=mask)
# #         final_frame = cv2.add(foreground, sky_region)
        
# #     elif SKY_MODE == "RED_MASK":
# #         # 创建一个纯红色的图层
# #         color_layer = np.zeros_like(original)
# #         color_layer[:] = [0, 0, 255] # BGR: Red
        
# #         # 将原图和红色混合 (原图 0.7 + 红色 0.3)
# #         blended_sky = cv2.addWeighted(original, 0.7, color_layer, 0.3, 0)
        
# #         # 抠图合并
# #         sky_region = cv2.bitwise_and(blended_sky, blended_sky, mask=mask_inv)
# #         foreground = cv2.bitwise_and(depth_vis, depth_vis, mask=mask)
# #         final_frame = cv2.add(foreground, sky_region)

# #     elif SKY_MODE == "BLUE_MASK":
# #         color_layer = np.zeros_like(original)
# #         color_layer[:] = [255, 0, 0] # BGR: Blue
# #         blended_sky = cv2.addWeighted(original, 0.7, color_layer, 0.3, 0)
        
# #         sky_region = cv2.bitwise_and(blended_sky, blended_sky, mask=mask_inv)
# #         foreground = cv2.bitwise_and(depth_vis, depth_vis, mask=mask)
# #         final_frame = cv2.add(foreground, sky_region)

# #     return final_frame

# # def run_video_pipeline():
# #     # ... (环境检查同上) ...
# #     if not os.path.exists(VIDEO_PATH): return

# #     orig_w, orig_h, orig_fps, _ = get_video_info(VIDEO_PATH)
    
# #     if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
# #     frames_dir = os.path.join(TEMP_DIR, "frames")
# #     output_frames_dir = os.path.join(TEMP_DIR, "output")
# #     os.makedirs(frames_dir, exist_ok=True)
# #     os.makedirs(output_frames_dir, exist_ok=True)

# #     # 1. 拆帧
# #     print("🎬 [1/3] 拆帧中...")
# #     subprocess.run(["ffmpeg", "-i", VIDEO_PATH, "-q:v", "2", os.path.join(frames_dir, "%05d.jpg")], 
# #                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# #     # 2. 推理
# #     print("🧠 [2/3] 深度推理中...")
# #     cmd = [
# #         "python", SCRIPT_PATH,
# #         "--input", frames_dir,
# #         "--output", output_frames_dir,
# #         "--pretrained", MODEL_PATH,
# #         "--maps", "--fp16"
# #     ]
# #     env = os.environ.copy()
# #     env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# #     subprocess.run(cmd, env=env, check=True)

# #     # 3. 融合与合成
# #     print(f"🎨 [3/3] 正在合成 (模式: {SKY_MODE})...")
    
# #     # 获取文件列表
# #     # 注意：infer.py 有时会生成 output/00001_depth_vis.png，有时会在子文件夹
# #     # 我们这里假设是平铺的，或者简单的 glob 匹配
# #     depth_files = sorted(glob.glob(os.path.join(output_frames_dir, "*_depth_vis.png")))
# #     # 原图文件
# #     orig_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

# #     if len(depth_files) == 0:
# #         # 尝试递归查找 (应对子文件夹情况 output/00001/depth_vis.png)
# #         depth_files = sorted(glob.glob(os.path.join(output_frames_dir, "**", "*_depth_vis.png"), recursive=True))

# #     if len(depth_files) != len(orig_files):
# #         print(f"⚠️ 警告: 深度图数量({len(depth_files)}) 与 原图数量({len(orig_files)}) 不一致，可能截断视频。")
# #         # 取最小长度
# #         min_len = min(len(depth_files), len(orig_files))
# #         depth_files = depth_files[:min_len]
# #         orig_files = orig_files[:min_len]

# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, orig_fps, (orig_w, orig_h))

# #     for d_path, o_path in tqdm(zip(depth_files, orig_files), total=len(depth_files)):
# #         # 调用融合函数
# #         frame = process_frame_fusion(d_path, o_path, (orig_w, orig_h))
# #         if frame is not None:
# #             video.write(frame)

# #     video.release()
# #     shutil.rmtree(TEMP_DIR)
# #     print(f"\n🎉 完成！输出文件: {OUTPUT_VIDEO_PATH}")

# # if __name__ == "__main__":
# #     run_video_pipeline()
# import os
# import subprocess
# import shutil
# import glob
# import cv2
# import numpy as np
# from tqdm import tqdm

# # ================= 配置区域 =================

# # 1. 输入视频路径
# VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1.mp4"

# # 2. 输出视频路径
# OUTPUT_VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1_depth_sky.mp4"

# # 3. 脚本与模型路径
# SCRIPT_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/infer.py"
# MODEL_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"

# # 4. 临时文件夹 (为了安全，建议换个新名字避免缓存干扰)
# TEMP_DIR = "./temp_video_inference_fixed"

# # 5. 天空处理模式: "ORIGINAL", "RED_MASK", "BLUE_MASK"
# SKY_MODE = "RED_MASK" 

# # ==========================================

# def get_video_info(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened(): raise ValueError(f"无法打开: {video_path}")
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#     return w, h, fps, frames

# def process_frame_fusion(depth_vis_path, original_img_path, target_size):
#     """
#     融合深度图和原图，处理天空
#     """
#     # 读取图片
#     depth_vis = cv2.imread(depth_vis_path)
#     original = cv2.imread(original_img_path)
    
#     if depth_vis is None:
#         print(f"❌ 无法读取深度图: {depth_vis_path}")
#         return None
#     if original is None:
#         print(f"❌ 无法读取原图: {original_img_path}")
#         return None

#     # 强制调整尺寸对齐
#     w, h = target_size
#     if depth_vis.shape[:2] != (h, w):
#         depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)
#     if original.shape[:2] != (h, w):
#         original = cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR)

#     # 提取天空掩码 (黑色区域为天空)
#     gray = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
#     mask_inv = cv2.bitwise_not(mask) # 255是天空

#     # 融合逻辑
#     if SKY_MODE == "ORIGINAL":
#         sky_region = cv2.bitwise_and(original, original, mask=mask_inv)
#         foreground = cv2.bitwise_and(depth_vis, depth_vis, mask=mask)
#         final_frame = cv2.add(foreground, sky_region)
        
#     elif SKY_MODE == "RED_MASK":
#         color_layer = np.zeros_like(original)
#         color_layer[:] = [0, 0, 255] # Red
#         blended_sky = cv2.addWeighted(original, 0.7, color_layer, 0.3, 0)
#         sky_region = cv2.bitwise_and(blended_sky, blended_sky, mask=mask_inv)
#         foreground = cv2.bitwise_and(depth_vis, depth_vis, mask=mask)
#         final_frame = cv2.add(foreground, sky_region)
        
#     elif SKY_MODE == "BLUE_MASK":
#         color_layer = np.zeros_like(original)
#         color_layer[:] = [255, 0, 0] # Blue
#         blended_sky = cv2.addWeighted(original, 0.7, color_layer, 0.3, 0)
#         sky_region = cv2.bitwise_and(blended_sky, blended_sky, mask=mask_inv)
#         foreground = cv2.bitwise_and(depth_vis, depth_vis, mask=mask)
#         final_frame = cv2.add(foreground, sky_region)
#     else:
#         final_frame = depth_vis # 默认不处理

#     return final_frame

# def run_video_pipeline():
#     if not os.path.exists(VIDEO_PATH): 
#         print(f"❌ 视频不存在: {VIDEO_PATH}")
#         return

#     # 获取原视频信息
#     orig_w, orig_h, orig_fps, _ = get_video_info(VIDEO_PATH)
    
#     # 清理并重建临时目录
#     if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
#     frames_dir = os.path.join(TEMP_DIR, "frames")
#     output_frames_dir = os.path.join(TEMP_DIR, "output")
#     os.makedirs(frames_dir, exist_ok=True)
#     os.makedirs(output_frames_dir, exist_ok=True)

#     # 1. 拆帧
#     print("🎬 [1/3] 拆帧中...")
#     subprocess.run(["ffmpeg", "-i", VIDEO_PATH, "-q:v", "2", os.path.join(frames_dir, "%05d.jpg")], 
#                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     # 2. 推理
#     print("🧠 [2/3] 深度推理中...")
#     # 注意：这里我们让 output 直接指向 output_frames_dir
#     # MoGe 会在 output_frames_dir 下生成 00001/ 00002/ 等文件夹
#     cmd = [
#         "python", SCRIPT_PATH,
#         "--input", frames_dir,
#         "--output", output_frames_dir,
#         "--pretrained", MODEL_PATH,
#         "--maps", "--fp16"
#     ]
#     env = os.environ.copy()
#     env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
#     subprocess.run(cmd, env=env, check=True)

#     # 3. 融合与合成 (🔥关键修改部分)
#     print(f"🎨 [3/3] 正在合成 (模式: {SKY_MODE})...")
    
#     # --- 🔥 修正的文件查找逻辑 ---
#     # 你的结构是: output/00001/depth_vis_scaled.png
#     # 我们使用通配符 */depth_vis_scaled.png 来匹配任意子文件夹
#     search_pattern = os.path.join(output_frames_dir, "*", "depth_vis_scaled.png")
#     depth_files = sorted(glob.glob(search_pattern))
    
#     # 原始图片是平铺的: frames/00001.jpg
#     orig_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

#     print(f"   🔍 扫描结果: 找到深度图 {len(depth_files)} 张, 原图 {len(orig_files)} 张")

#     # 再次校验
#     if len(depth_files) == 0:
#         print("❌ 依然未找到深度图！请检查 search_pattern 是否正确。")
#         print(f"   当前搜索路径: {search_pattern}")
#         # 尝试打印一下 output 目录结构帮助调试
#         print("   Temp目录结构预览:")
#         os.system(f"ls -R {output_frames_dir} | head -n 10")
#         return

#     # 对齐数量
#     min_len = min(len(depth_files), len(orig_files))
#     depth_files = depth_files[:min_len]
#     orig_files = orig_files[:min_len]

#     # 初始化视频写入
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, orig_fps, (orig_w, orig_h))

#     for d_path, o_path in tqdm(zip(depth_files, orig_files), total=len(depth_files)):
#         frame = process_frame_fusion(d_path, o_path, (orig_w, orig_h))
#         if frame is not None:
#             video.write(frame)

#     video.release()
#     print(f"\n🎉 完美！视频已生成: {OUTPUT_VIDEO_PATH}")
    
#     # 如果成功，清理临时文件
#     shutil.rmtree(TEMP_DIR)

# if __name__ == "__main__":
#     run_video_pipeline()

import os
import cv2
import torch
import numpy as np
import subprocess
import shutil
import glob
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# ================= 🎛️ 参数配置区域 =================

# 1. 输入输出路径
VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Moge2/ade7b1b86c91ecba5f688b59db6cd9a1.mp4"
OUTPUT_VIDEO_PATH = "/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/ade7b1b86c91ecba5f688b59db6cd9a1_dav2_smooth.mp4"

# 2. 模型路径
MODEL_PATH = '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth'
ENCODER = 'vitl'

# 3. 天空处理参数 (关键！)
# 阈值 (0-255): 小于这个值的深度会被认为是天空。值越大，被替换的区域越多。
# DA-V2 的天空通常非常暗，建议 10-25 之间试一试。
SKY_THRESHOLD = 20 

# 过渡柔和度 (必须是奇数): 值越大，边缘越模糊，过渡越自然。建议 21, 31, 51 等。
BLUR_KSIZE = 31  

# 天空遮罩颜色 (BGR): 这里设为淡蓝色 [230, 216, 173] (SkyBlue)
# 或者淡红色 [100, 100, 255]
SKY_TINT_COLOR = [100, 100, 255] 

# 深度图配色
COLORMAP = cv2.COLORMAP_INFERNO

# ==================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def load_model():
    print(f"Loading Depth Anything V2 ({ENCODER})...")
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"无法打开视频: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, frames

def process_frame_smooth(raw_img, depth, target_size):
    """
    核心渲染函数：带有羽化过渡的天空融合
    """
    w, h = target_size
    
    # 1. 预处理原图 (Resize)
    if raw_img.shape[:2] != (h, w):
        raw_img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. 预处理深度图 (Resize + Normalize)
    # DA-V2 推理出来的大小可能跟原图不一样，先插值回原图大小
    # 注意：在 float 状态下插值精度更高
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 归一化到 0-255
    d_min = depth_resized.min()
    d_max = depth_resized.max()
    depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # 3. 生成深度热力图 (Layer A)
    depth_color = cv2.applyColorMap(depth_uint8, COLORMAP)

    # 4. 生成天空背景层 (Layer B)
    # 创建一个纯色层
    color_overlay = np.zeros_like(raw_img)
    color_overlay[:] = SKY_TINT_COLOR
    # 原图与颜色混合：原图 70% + 颜色 30%
    sky_layer = cv2.addWeighted(raw_img, 0.7, color_overlay, 0.3, 0)

    # 5. 生成遮罩 (Mask Generation)
    # 找出深度值小于阈值的区域 (天空/远处)
    # mask 为 255 的地方是天空
    _, mask = cv2.threshold(depth_uint8, SKY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # 6. ✨ 关键步骤：边缘羽化 (Gaussian Blur) ✨
    # 将二值 Mask 变成平滑的 Alpha 通道
    mask_float = mask.astype(np.float32) / 255.0 # 转为 0.0 - 1.0
    # 使用高斯模糊处理边缘
    mask_blurred = cv2.GaussianBlur(mask_float, (BLUR_KSIZE, BLUR_KSIZE), 0)
    
    # 扩展 mask 维度以匹配图像通道 [H, W] -> [H, W, 3]
    mask_3ch = np.stack([mask_blurred] * 3, axis=-1)

    # 7. Alpha 混合 (Compositing)
    # 公式: Result = Sky * Alpha + Depth * (1 - Alpha)
    # 当 Mask 为 1 (天空) 时，显示 sky_layer
    # 当 Mask 为 0 (近处) 时，显示 depth_color
    # 当 Mask 为 0.5 (边缘) 时，两者混合
    final_frame = sky_layer * mask_3ch + depth_color * (1.0 - mask_3ch)

    return final_frame.astype(np.uint8)

def run_video_inference():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误: 找不到输入视频: {VIDEO_PATH}")
        return

    # 1. 加载模型
    model = load_model()
    orig_w, orig_h, orig_fps, _ = get_video_info(VIDEO_PATH)
    
    # 2. 准备临时目录拆帧
    TEMP_DIR = "./temp_dav2_smooth"
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    frames_dir = os.path.join(TEMP_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("🎬 [1/3] 正在拆帧...")
    subprocess.run(["ffmpeg", "-i", VIDEO_PATH, "-q:v", "2", os.path.join(frames_dir, "%05d.jpg")], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    img_list = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    # 3. 处理并写入
    print(f"🧠 [2/3] 开始生成平滑过渡视频 (Threshold={SKY_THRESHOLD}, Blur={BLUR_KSIZE})...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, orig_fps, (orig_w, orig_h))
    
    for img_path in tqdm(img_list):
        raw_image = cv2.imread(img_path)
        
        # 推理 (使用较小的 input_size 保证速度，结果会 Resize 回去)
        depth = model.infer_image(raw_image, input_size=518)
        
        # 调用平滑融合函数
        final_frame = process_frame_smooth(raw_image, depth, (orig_w, orig_h))
        
        video_writer.write(final_frame)

    video_writer.release()
    shutil.rmtree(TEMP_DIR)
    
    print(f"\n🎉 完美！平滑融合视频已生成: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run_video_inference()