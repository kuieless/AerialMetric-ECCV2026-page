import sys
import argparse  # <--- 就是少了这一行
# 1. 确保能找到 moge_lora_core.py
# 如果 core 文件在当前目录，直接 import 即可
# 如果在其他目录，请取消下面注释并修改路径
sys.path.append("/home/szq/moge2/MoGe/moge/scripts")
import argparse  # <--- 就是少了这一行
from a_infer_lora96 import run_inference_pipeline

# ================= ⚙️ 用户配置区域 =================

# 实验名称 (用于区分输出目录)
EXP_NAME = "Final_Run_Checkpoint_2000"

# 输入数据集列表
INPUT_ROOTS = [
    "/data1/szq/Val/", 
    # "/home/szq/moge2/DJI-self2-final/" # 可以添加更多
]

# 输出根目录
OUTPUT_ROOT = f"/data1/szq/Inference_Results_LoRA_{EXP_NAME}"

# LoRA 模型设置
LORA_CONFIG = "/home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json"
LORA_WEIGHT = "/home/szq/moge2/MoGe/workspace/final-fintune2-130-lora-final6-64r/checkpoint/00002000.pt"

# 推理参数
PARAMS = {
    "sampling_ratio": 0.01,  # 1% 的图片 (测试用，正式跑改 1.0)
    "resize": None,          # 长边 resize 到 1024 推理，然后还原
    "device": "cuda"
}

# ================= 🚀 执行 =================

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="LoRA Inference Pipeline")
    
#     # 基础路径参数
#     parser.add_argument("--input", default=None, help="输入根目录")
#     parser.add_argument("--output", default=None, help="输出根目录")
    
#     # LoRA 专属参数
#     parser.add_argument("--config", required=True, help="LoRA config.json 路径")
#     parser.add_argument("--weight", required=True, help="LoRA .pt 权重路径")
    
#     # 推理参数
#     parser.add_argument("--resize", type=int, default=0, help="Resize大小 (0代表原图)")
#     parser.add_argument("--ratio", type=float, default=0.01, help="采样率")
    
#     args = parser.parse_args()

#     # 1. 实验配置
#     # 如果命令行没传 input，就用默认的 SEARCH_ROOTS (为了兼容性)
#     input_roots = [args.input] if args.input else SEARCH_ROOTS
#     output_root = args.output if args.output else OUTPUT_ROOT_BASE
    
#     # 2. 处理 resize (0 -> None) 防止 OpenCV 崩溃
#     real_resize = args.resize
#     if real_resize <= 0:
#         real_resize = None

#     print(f"🚀 启动 LoRA 实验")
#     print(f"   Input:  {input_roots}")
#     print(f"   Output: {output_root}")
#     print(f"   Config: {args.config}")
#     print(f"   Weight: {args.weight}")
#     print(f"   Resize: {real_resize}")
    
#     # 3. 调用核心函数
#     run_inference_pipeline(
#         input_roots=input_roots,
#         output_root=output_root,
#         lora_config=args.config,
#         lora_weight=args.weight,
#         sampling_ratio=args.ratio,
#         resize=real_resize
#     )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Inference Pipeline")
    parser.add_argument("--input", default=None, help="输入根目录")
    parser.add_argument("--output", default=None, help="输出根目录")
    parser.add_argument("--config", required=True, help="LoRA config.json 路径")
    parser.add_argument("--weight", required=True, help="LoRA .pt 权重路径")
    parser.add_argument("--resize", type=int, default=0, help="Resize大小 (0代表原图)")
    parser.add_argument("--ratio", type=float, default=0.01, help="采样率")
    
    # 🔥 新增 Batch Size 参数
    parser.add_argument("--batch_size", type=int, default=4, help="并行推理数量，根据显存调整")
    
    args = parser.parse_args()

    input_roots = [args.input] if args.input else []
    output_root = args.output
    
    real_resize = args.resize if args.resize > 0 else None

    print(f"🚀 启动 LoRA 实验 (Batch Size: {args.batch_size})")
    
    run_inference_pipeline(
        input_roots=input_roots,
        output_root=output_root,
        lora_config=args.config,
        lora_weight=args.weight,
        sampling_ratio=args.ratio,
        resize=real_resize,
        batch_size=args.batch_size # 🔥 传给核心
    )