import cv2
import numpy as np
import sys

# 随便找一张你的 depth.png 的路径填进去测试
FILE_PATH = "/home/data1/szq/Megadepth/Benchmark-final2/moge-eva/DDAD/val/000007/CAMERA_06/depth.png" 

def check_depth_png(path):
    # 1. 以原始模式读取 (不改变位深)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"❌ 无法读取文件: {path}")
        return

    print(f"📂 文件: {path}")
    print(f"📏 形状: {img.shape}")
    print(f"💾 类型 (dtype): {img.dtype}")
    print(f"⬇️ 最小值: {img.min()}")
    print(f"⬆️ 最大值: {img.max()}")
    
    # 常见深度图判断逻辑
    if img.dtype == 'uint16':
        print("\n🧐 推测: 这是一个 16-bit 深度图。")
        print("   通常 1 unit = 1 mm (除以 1000 得到米)")
        print("   或者 1 unit = 1/256 m (除以 256 得到米)")
    elif img.dtype == 'uint8':
        print("\n🧐 推测: 这是一个 8-bit 深度图 (精度较低，主要用于可视化)。")
    else:
        print("\n🧐 推测: 未知格式，可能是 float 或者 encode 过的。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]
    check_depth_png(FILE_PATH)