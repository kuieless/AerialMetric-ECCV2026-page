import os
import subprocess
import glob
import matplotlib
# 服务器环境必须设置 Agg 后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ================= 配置区域 (修改这里) =================

# 1. 脚本和模型路径 (已按你要求修改)
SCRIPT_PATH = "/home/szq/moge2/MoGe/moge/scripts/infer.py"
MODEL_PATH = "/home/szq/moge2/MoGe/vitl-normal.pt"

# 2. 输入：可以是一张图片的路径，也可以是一个包含图片的文件夹路径
# 例如: "/home/data/test.jpg" 或者 "/home/data/my_images_folder"
INPUT_PATH = "/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-9k/BC2/000071/depth.npy" 

# 3. 输出：结果保存的文件夹
OUTPUT_DIR = "/data1/szq/data/becnmarkdata2/Val-Results-moge2-head-9k/BC2/000071/images_output"

# ================= 可视化工具类 (保持不变) =================

class ServerDepthVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_and_save(self, depth_npy_path: str, rgb_path: str = None):
        try:
            depth_data = np.load(depth_npy_path)
            
            rgb_image = None
            if rgb_path and os.path.exists(rgb_path):
                rgb_image = np.array(Image.open(rgb_path))

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            ax_rgb, ax_depth = axes

            if rgb_image is not None:
                ax_rgb.imshow(rgb_image)
                ax_rgb.set_title("Source Image")
            else:
                ax_rgb.text(0.5, 0.5, 'No RGB', ha='center', va='center')
            ax_rgb.axis('off')

            valid_depth = depth_data[depth_data > 0]
            if len(valid_depth) > 0:
                vmin = np.nanquantile(valid_depth, 0.01)
                vmax = np.nanquantile(valid_depth, 0.99)
            else:
                vmin, vmax = 0, 10
            
            im = ax_depth.imshow(depth_data, cmap='Spectral', vmin=vmin, vmax=vmax)
            ax_depth.set_title("Depth Map (m)")
            ax_depth.axis('off')

            cbar = fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)
            cbar.set_label('Depth Value')

            basename = os.path.splitext(os.path.basename(depth_npy_path))[0]
            save_path = os.path.join(self.output_dir, f"{basename}_vis.png")
            
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> 可视化已保存: {os.path.basename(save_path)}")
            
        except Exception as e:
            print(f"    [!] 可视化失败: {depth_npy_path}, 错误: {e}")

# ================= 执行逻辑 =================

def run_inference_and_vis():
    # 0. 检查路径
    if not os.path.exists(SCRIPT_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ 错误: 找不到 infer.py 脚本或模型文件，请检查路径。")
        return
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 输入路径不存在: {INPUT_PATH}")
        return

    # 1. 组装命令
    env = os.environ.copy()
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    
    cmd = [
        "python", SCRIPT_PATH,
        "--input", INPUT_PATH,
        "--output", OUTPUT_DIR,
        "--pretrained", MODEL_PATH,
        "--maps",     # 必须加这个，才有 npy 输出
        # "--fp16"    # 如果显存够，建议取消注释以加速
    ]

    print(f"🚀 开始处理...")
    print(f"   输入: {INPUT_PATH}")
    print(f"   输出: {OUTPUT_DIR}")

    # 2. 运行推理 (MoGe)
    try:
        subprocess.run(cmd, env=env, check=True)
        print("✅ 推理完成，开始生成可视化图...")
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理出错，代码: {e.returncode}")
        return

    # 3. 运行可视化
    # 策略：扫描 OUTPUT_DIR 里所有的 .npy 文件，然后尝试去匹配输入图片
    vis_output_dir = os.path.join(OUTPUT_DIR, "vis_result")
    visualizer = ServerDepthVisualizer(vis_output_dir)

    # 递归查找输出目录下的所有 .npy
    npy_files = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    
    if not npy_files:
        print("⚠️  警告: 在输出目录没找到 .npy 文件，可能推理未生成 maps。")
        return

    # 确定输入图片的搜索目录
    # 如果 INPUT_PATH 是文件，原本图片就在 INPUT_PATH 里
    # 如果 INPUT_PATH 是目录，图片就在那个目录里
    if os.path.isfile(INPUT_PATH):
        input_search_dir = os.path.dirname(INPUT_PATH)
        specific_image_name = os.path.basename(INPUT_PATH) # 记住这个文件名
    else:
        input_search_dir = INPUT_PATH
        specific_image_name = None

    for npy_path in npy_files:
        # 这里的逻辑是尝试根据 npy 的文件名反推图片名
        # MoGe 输出通常是: filename_depth.npy 或者 filename.npy
        npy_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        # 简单的匹配逻辑：去 input 目录找同名文件
        found_img_path = None
        
        # 如果用户只输了一个图片，直接用那个图片就行（防止文件名变了匹配不上）
        if specific_image_name:
             found_img_path = INPUT_PATH
        else:
            # 如果是文件夹，尝试匹配
            # 这里的匹配逻辑比较宽泛，因为MoGe可能会加后缀
            # 我们尝试去掉 "_depth" 后缀再找
            clean_name = npy_name.replace("_depth", "").replace("_pred", "")
            
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.bmp']:
                test_path = os.path.join(input_search_dir, clean_name + ext)
                if os.path.exists(test_path):
                    found_img_path = test_path
                    break
        
        # 执行画图
        visualizer.process_and_save(npy_path, found_img_path)

    print(f"🎉 全部完成！结果保存在: {vis_output_dir}")

if __name__ == "__main__":
    run_inference_and_vis()