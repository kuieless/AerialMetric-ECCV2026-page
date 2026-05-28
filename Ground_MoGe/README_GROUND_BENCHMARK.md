# Ground Benchmark

This directory contains the ground metric-depth benchmark code used with MoGe-2.

The public entry point is:

```text
moge/scripts/eval_baseline.py
```

## Benchmarks

The local release config evaluates:

```text
NYUv2
KITTI
ETH3D
iBims-1
DDAD
DIODE
HAMMER
```

Config:

```text
configs/eval/ground_metric_benchmarks_local.json
```

The config currently points to:

```text
/data1/szq/Val/eval
```

If the datasets are stored elsewhere, edit the `path` fields in the config.

## Run Evaluation

Evaluate a local MoGe-2 checkpoint:

```bash
cd Ground_MoGe

CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  moge/scripts/eval_baseline.py \
  --baseline baselines/moge2_metric.py \
  --config configs/eval/ground_metric_benchmarks_local.json \
  --output eval_output_release/moge2_ground.json \
  --checkpoint /path/to/model_checkpoint.pt \
  --resolution_level 9 \
  --fp16 \
  --device cuda:0
```

Use GT intrinsics:

```bash
cd Ground_MoGe

CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  moge/scripts/eval_baseline.py \
  --baseline baselines/moge2_metric.py \
  --config configs/eval/ground_metric_benchmarks_local.json \
  --output eval_output_release/moge2_ground_gt_intr.json \
  --checkpoint /path/to/model_checkpoint.pt \
  --resolution_level 9 \
  --fp16 \
  --device cuda:0 \
  --oracle
```

## Outputs

The main output is a JSON file containing aggregate metrics and grouped statistics.

`eval_output*/`, checkpoints, and local datasets are ignored by git.
