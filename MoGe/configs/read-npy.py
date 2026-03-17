# import numpy as np
# import matplotlib.pyplot as plt

# # ================= 配置 =================
# # 替换为你刚才生成的任意一个 npy 文件路径
# NPY_PATH = "/data1/szq/Syn-train/city_ouputfov64/2a5ccd6ba40482e9273a12800c083d04d6589770/depth.npy" 
# # =======================================

# def check_depth_sky():
#     # 1. 读取数据
#     if not os.path.exists(NPY_PATH):
#         print("❌ 文件路径不存在，请修改 NPY_PATH")
#         return

#     depth = np.load(NPY_PATH)
#     h, w = depth.shape

#     print(f"📊 数据统计: {NPY_PATH}")
#     print(f"   尺寸: {w}x{h}")
#     print(f"   最小值: {np.min(depth):.4f} m")
#     print(f"   最大值: {np.max(depth):.4f} m")
#     print(f"   平均值: {np.mean(depth):.4f} m")

#     # 2. 采样顶部区域 (通常天空在图片最上方)
#     # 取顶部 20 行像素
#     top_strip = depth[0:20, :]
    
#     # 找顶部区域出现频率最高的数值 (Mode)
#     vals, counts = np.unique(top_strip, return_counts=True)
#     most_common_val = vals[np.argmax(counts)]
    
#     print(f"\n🌤️ 天空区域分析 (Top 20 rows):")
#     print(f"   出现最多的值 (可能是天空): {most_common_val:.4f} m")
    
#     # 3. 判断逻辑
#     if most_common_val > 600:
#         print("\n📝 结论: 你的天空看起来是 [最大距离] (约 655.35m)。")
#         print("   👉 在训练时，你需要把 > 600 的区域 Mask 掉 (设为 NaN)。")
#     elif most_common_val == 0:
#         print("\n📝 结论: 你的天空看起来是 [0]。")
#         print("   👉 在训练时，你需要把 == 0 的区域 Mask 掉。")
#     else:
#         print("\n📝 结论: 天空值不明显，可能是俯视视角或室内场景？建议看图确认。")

#     # 4. 可视化确认
#     plt.figure(figsize=(10, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.title("Depth Map (Meters)")
#     im = plt.imshow(depth, cmap='inferno')
#     plt.colorbar(im, label='Depth (m)')
    
#     plt.subplot(1, 2, 2)
#     plt.title("Histogram")
#     plt.hist(depth.flatten(), bins=100, log=True)
#     plt.xlabel("Depth (m)")
    
#     plt.tight_layout()
#     # 如果你在服务器上没有显示屏，可以保存图片
#     plt.savefig("check_sky_result.png")
#     print("\n🖼️ 已保存可视化结果到: check_sky_result.png (请下载查看)")

# if __name__ == "__main__":
#     import os
#     check_depth_sky()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_depth_map(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    # 1. 读取图像 (必须使用 IMREAD_UNCHANGED 以保持 16-bit 深度)
    depth_png = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if depth_png is None:
        print("❌ 错误: 无法读取图像，文件可能损坏或格式不支持。")
        return

    # 2. 基本信息
    h, w = depth_png.shape
    dtype = depth_png.dtype
    
    # 排除 0 值（通常 0 代表无效深度或天空）
    valid_mask = depth_png > 0
    if np.sum(valid_mask) == 0:
        print("⚠️ 警告: 图像全为 0，无法分析。")
        return
        
    valid_pixels = depth_png[valid_mask]
    
    min_val = np.min(valid_pixels)
    max_val = np.max(valid_pixels)
    mean_val = np.mean(valid_pixels)
    median_val = np.median(valid_pixels)

    print(f"--- 深度图分析报告 ---")
    print(f"📂 文件: {os.path.basename(file_path)}")
    print(f"📐 尺寸: {w}x{h}")
    print(f"💾 数据类型: {dtype} (通常应为 uint16)")
    print(f"----------------------")
    print(f"📉 最小值 (非0): {min_val}")
    print(f"📈 最大值: {max_val}")
    print(f"📊 平均值: {mean_val:.2f}")
    print(f"🎯 中位数: {median_val:.2f}")
    print(f"----------------------")

    # 3. 推断单位逻辑
    print("🔍 单位推断:")
    
    if dtype != 'uint16':
        print(f"⚠️ 注意: 数据类型是 {dtype}，大多数标准深度图是 uint16。如果这是 uint8，通常代表相对深度(0-255)。")
    
    # 假设 1: 单位是 毫米 (mm) -> value / 1000.0 = meters
    meters_mm_assumption = median_val / 1000.0
    
    # 假设 2: 单位是 厘米 (cm) -> value / 100.0 = meters
    meters_cm_assumption = median_val / 100.0

    # 假设 3: 某些特殊数据集 (如 TartanAir) 可能用特殊的缩放
    
    print(f"👉 假设单位是 [毫米] (常见):")
    print(f"   场景中位数深度约为: {meters_mm_assumption:.2f} 米")
    print(f"   场景最大深度约为: {max_val / 1000.0:.2f} 米")
    
    if 0.5 < meters_mm_assumption < 100:
        print("   ✅ 这看起来非常合理 (通常室内外场景深度在 1m 到 100m 之间)。")
    else:
        print("   ❓ 这看起来不太合理 (太近或太远)。")

    print(f"\n👉 假设单位是 [厘米]:")
    print(f"   场景中位数深度约为: {meters_cm_assumption:.2f} 米")
    
    # 4. 可视化 (可选)
    # 将深度归一化以便显示
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_png, cmap='inferno')
    plt.colorbar(label=f'Raw Pixel Value ({dtype})')
    plt.title(f'Depth Map Visualization\nMax Value: {max_val}')
    plt.show()

# 你的文件路径
file_path = "/data1/yzy/MoGev2/data/train/MVS-Synth/0000/0000/depth.png"
analyze_depth_map(file_path)