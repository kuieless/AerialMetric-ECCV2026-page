# Third-Party Depth Baselines

This directory contains local source copies and a unified evaluation entry point for:

```text
UniDepth
DepthPro
ZoeDepth
```

The public CLI is:

```text
Depth_Baselines/eval_depth_baseline_cli.py
```

It evaluates the selected model with the ground benchmark code in `Ground_MoGe/`.

## Ground Evaluation

UniDepth:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  Depth_Baselines/eval_depth_baseline_cli.py \
  --model unidepth \
  --gpu 0 \
  --fp16
```

DepthPro:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  Depth_Baselines/eval_depth_baseline_cli.py \
  --model depthpro \
  --gpu 0 \
  --fp16
```

ZoeDepth:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  Depth_Baselines/eval_depth_baseline_cli.py \
  --model zoedepth \
  --gpu 0
```

Use GT intrinsics:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  Depth_Baselines/eval_depth_baseline_cli.py \
  --model unidepth \
  --gpu 0 \
  --oracle \
  --fp16
```

By default, the CLI uses:

```text
Ground_MoGe/configs/eval/ground_metric_benchmarks_local.json
```

Outputs are written to:

```text
eval_output_release/<model>_ground.json
```

## Model Sources

Local source directories:

```text
Depth_Baselines/UniDepth
Depth_Baselines/ml-depth-pro
Depth_Baselines/ZoeDepth
```

Checkpoints are intentionally not included in git. DepthPro can use a local checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  Depth_Baselines/eval_depth_baseline_cli.py \
  --model depthpro \
  --checkpoint /path/to/depth_pro.pt \
  --gpu 0 \
  --fp16
```

UniDepth and ZoeDepth can load their default pretrained resources when network and cache access are available.
