import sys
import argparse 
sys.path.append("/home/szq/moge2/MoGe/moge/scripts")
from a_infer_lora96_norm import run_inference_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Inference Pipeline")
    parser.add_argument("--input", required=True, help="输入根目录 (如 /data1/szq/Val/Oblique-norm)")
    parser.add_argument("--output", required=True, help="输出根目录 (如 /data1/szq/Result/Oblique-norm)")
    parser.add_argument("--config", required=True, help="LoRA config.json 路径")
    parser.add_argument("--weight", required=True, help="LoRA .pt 权重路径")
    parser.add_argument("--resize", type=int, default=1024, help="Resize大小 (0代表原图)")
    parser.add_argument("--ratio", type=float, default=1.0, help="采样率 (1.0表示全量)")
    parser.add_argument("--batch_size", type=int, default=4, help="并行推理数量，根据显存调整")
    
    args = parser.parse_args()

    input_roots = [args.input]
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
        batch_size=args.batch_size
    )
'''
python /home/szq/moge2/MoGe/moge/scripts/code-final/a-infer_lora96-norm.py \
  --input /data1/szq/Val/Oblique-norm \
  --output /data1/szq/Val/Oblique-norm-results \
  --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json \
  --weight /data1/szq/workspace/lora-batch96-192-with-UElr2/checkpoints/00001000.pt \
  --resize 0 \
  --ratio 1.0 \
  --batch_size 8

python /home/szq/moge2/MoGe/moge/scripts/code-final/a-infer_lora96-norm.py \
  --input /data1/szq/Val/Bench-norm \
  --output /data1/szq/Val/Bench-norm-results \
  --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json \
  --weight /data1/szq/workspace/lora-batch96-192-with-UElr2/checkpoints/00001000.pt \
  --resize 0 \
  --ratio 1.0 \
  --batch_size 8
'''