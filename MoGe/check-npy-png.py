import numpy as np
from moge.utils.io import read_depth

# 找一个具体的样本路径
npy_path = "/home/data1/szq/Megadepth/TrainingData_Final_MoGe-6-all/dj3/dj3-0-000003/depth.npy"
png_path = "/home/data1/szq/Megadepth/TrainingData_MoGe_PNG/dj3/dj3-0-000003/depth.png"

d_npy = np.load(npy_path)
d_png = read_depth(png_path)

# 计算 100米以上的误差
mask = (d_npy > 100)
diff = np.abs(d_npy[mask] - d_png[mask])

print(f"最大误差: {diff.max()}")
print(f"平均误差: {diff.mean()}")