# AerialMetric

This is the repository that contains source code for the [AerialMetric](https://kuieless.github.io/AerialMetric/).

## Benchmark Entry Points

Run commands from the repository root unless the command explicitly starts with `cd`.

```bash
cd /path/to/AerialMetric
```

The repository is organized into three benchmark interfaces:

| Directory | Purpose |
|---|---|
| `MoGe/` | Aerial benchmark for AerialMetric MoGe variants |
| `Ground_MoGe/` | Ground benchmark for MoGe-2 checkpoints |
| `Depth_Baselines/` | Ground benchmark wrappers for UniDepth, DepthPro, and ZoeDepth |

Dataset layout requirements are documented in `DATA_ORGANIZATION.md`.

### 1. Aerial Benchmark

Use this command for AerialMetric MoGe variants on Decoupled, Oblique, and Wild.
Use `--intrinsics_mode none` or `auto` with the standard layout, and
`--intrinsics_mode load` with the norm-style Decoupled/Oblique inputs that include
`meta.json`. Wild stays on the standard layout and is handled with `auto`.

Supported `--model_type` values:

```text
full, freezevit, scaleheadonly, lora64, lora96, lora128
```

Example:

```bash
cd /path/to/AerialMetric

CUDA_VISIBLE_DEVICES=7 conda run -n moge310 python \
  MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint /path/to/Moge2-Aerial.pt \
  --lora_config MoGe/configs/Final_train/config-lora-all.json \
  --decoupled_input /path/to/decoupled \
  --decoupled_gt /path/to/decoupled-norm \
  --decoupled_csv_dir /path/to/decoupled \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --wild_input /path/to/Wild \
  --wild_gt /path/to/Wild \
  --output_dir /path/to/eval_outputs \
  --gpu 7 \
  --resize 0 \
  --batch_size 8 \
  --intrinsics_mode none \
  --mask_mode none \
  --cleanup_intermediate
```

For non-LoRA checkpoints, use `--model_type full`, `head`, or `neck` and omit `--lora_config`.

More details:

```text
MoGe/README_AERIAL_BENCHMARK.md
```

The dataset layouts for both modes are documented in `DATA_ORGANIZATION.md`.

### 2. Ground Benchmark For MoGe-2

Use these commands to evaluate MoGe-2 checkpoints on the ground metric-depth benchmark:

```text
NYUv2, KITTI, ETH3D, iBims-1, DDAD, DIODE, HAMMER
```

This command runs from inside `Ground_MoGe/`:

```bash
cd /path/to/AerialMetric/Ground_MoGe

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

Use GT intrinsics by adding:

```bash
--oracle
```

For the LoRA ground path, use the dedicated wrapper and LoRA baseline:

```bash
cd /path/to/AerialMetric/Ground_MoGe

CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  moge/scripts/eval_baselinelora.py \
  --baseline baselines/moge2_lora.py \
  --lora_config /path/to/config-lora-all.json \
  --lora_weight /path/to/lora_checkpoint.pt \
  --lora_rank 96 \
  --resolution_level 9 \
  --config configs/eval/ground_metric_benchmarks_local.json \
  --output eval_output_release/moge2_ground_lora.json \
  --oracle
```

The default local config points to:

```text
/data1/szq/Val/eval
```

Edit `Ground_MoGe/configs/eval/ground_metric_benchmarks_local.json` if your dataset root is different.

For a 5% smoke test, use a config where each dataset sets `subset: 20`.

More details:

```text
Ground_MoGe/README_GROUND_BENCHMARK.md
```

### 3. Third-Party Depth Baselines

Use this wrapper to evaluate UniDepth, DepthPro, or ZoeDepth on the same ground benchmark.

Run from the repository root:

```bash
cd /path/to/AerialMetric
```

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

Use GT intrinsics by adding:

```bash
--oracle
```

DepthPro checkpoints are not included in git. To use a local DepthPro checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  Depth_Baselines/eval_depth_baseline_cli.py \
  --model depthpro \
  --checkpoint /path/to/depth_pro.pt \
  --gpu 0 \
  --fp16
```

More details:

```text
Depth_Baselines/README_DEPTH_BASELINES.md
```

If you find Nerfies useful for your work please cite:
```
# TBD
@inproceedings{aerialmetric,
  title     = {AerialMetric: Benchmarking and Adapting UAV Monocular Metric Depth Estimation in the Real World},
  author    = {Anonymous},
  booktitle = {Anonymous},
  year      = {2026}
}
```

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
