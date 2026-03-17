import sys

# 确保能找到 infer_core.py
sys.path.append("/home/szq/moge2/MoGe/moge/scripts")

from infer_core_norm import run_base_inference_pipeline

# ================= ⚙️ 用户配置区域 =================

EXP_NAME = "Base_Model_V2_Original_Size"

# 输入数据集列表
INPUT_ROOTS = [
    "/data1/szq/Wild", 
]

# 输出根目录
OUTPUT_ROOT = f"/data1/szq/Inference_Results_{EXP_NAME}"

# 模型路径
MODEL_PATH = "/home/szq/moge2/MoGe/vitl-normal.pt"

# 推理参数
PARAMS = {
    "version": "v2",         # 模型版本 v1/v2
    "sampling_ratio": 0.01,  # 100% 全量推理
    "resize": None,          # None 表示原图尺寸 (自动对齐14)
    "device": "cuda"
}

# ================= 🚀 执行 =================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Model Inference Pipeline")
    parser.add_argument("--input", default=None, help="输入根目录")
    parser.add_argument("--output", default=None, help="输出根目录")
    parser.add_argument("--model", default="/home/szq/moge2/MoGe/vitl-normal.pt", help="模型路径")
    
    parser.add_argument("--resize", type=int, default=0, help="Resize大小 (0代表原图)")
    parser.add_argument("--ratio", type=float, default=0.01, help="采样率")
    
    # 🔥 新增 Batch Size 参数
    parser.add_argument("--batch_size", type=int, default=4, help="并行推理数量，根据显存调整")
    
    args = parser.parse_args()

    # 实验配置
    input_roots = [args.input] if args.input else INPUT_ROOTS
    output_root = args.output if args.output else OUTPUT_ROOT
    model_path = args.model
    
    # ================= 🔥 核心修复逻辑 🔥 =================
    real_resize = args.resize
    if real_resize <= 0:
        real_resize = None  
    # ===================================================

    # 参数配置
    params = PARAMS.copy()
    params['sampling_ratio'] = args.ratio
    params['resize'] = real_resize 
    params['batch_size'] = args.batch_size # 🔥 传给核心

    print(f"🚀 启动 Base Model 实验 (Batch Size: {args.batch_size})")
    print(f"   Input:  {input_roots}")
    print(f"   Output: {output_root}")
    print(f"   Resize: {real_resize} (Shell传入的是 {args.resize})")
    
    run_base_inference_pipeline(
        input_roots=input_roots,
        output_root=output_root,
        model_path=model_path,
        **params
    )
