# Aerial Benchmark

This repository provides a unified inference and evaluation entry point for aerial metric depth benchmark experiments based on MoGe.

The recommended public interface is:

```text
moge/scripts/code-final/aerial_eval_cli.py
```

It supports six model variants:

| Model type | Description |
|---|---|
| `full` | Full-parameter / standard checkpoint |
| `head` | Head-tuned checkpoint |
| `neck` | Neck-tuned checkpoint |
| `lora64` | LoRA checkpoint with rank 64 and alpha 128 |
| `lora96` | LoRA checkpoint with rank 96 and alpha 192 |
| `lora128` | LoRA checkpoint with rank 128 and alpha 256 |

## Environment

Create or activate the project environment before running evaluation.

Example:

```bash
conda activate moge310
```

The CLI should be launched from the repository environment where MoGe, PyTorch, OpenCV, PEFT, and the project dependencies are available.

## Pipeline

The CLI runs the complete benchmark pipeline:

```text
input images
  -> inference
  -> extract nested depth.npy files
  -> evaluate Bench / Oblique / Wild
  -> optionally delete intermediate predictions
```

Output layout:

```text
<output_dir>/<model_type>/<checkpoint_name>/
  Infer/<Dataset>/<Scene>/<SampleID>/depth.npy
  Extracted/<Dataset>/<Scene>/<SampleID>.npy
  Extracted/<Dataset>/Eval_Report_*.txt
  run.log
```

If `--cleanup_intermediate` is used, the CLI removes `Infer/` and extracted `.npy` files after reports are generated. Text and CSV reports are kept.

## Data Layout

The inference scanner supports these image layouts:

```text
Scene/SampleID/image.jpg
Scene/SampleID/image.png
Scene/image/*.jpg
Scene/images/*.jpg
Scene/rgb/*.jpg
Scene/rgbs/*.jpg
Scene/color/*.jpg
```

Bench evaluation supports:

```text
Bench GT:
  Scene/SampleID/depth.npy

Bench CSV metadata:
  final_dataset_campus.csv
  final_dataset_factory.csv
  final_dataset_farm.csv
  final_dataset_grass.csv
```

If the CSV files are stored separately from the GT root, pass `--bench_csv_dir`.

## Common Options

| Option | Description |
|---|---|
| `--model_type` | One of `full`, `head`, `neck`, `lora64`, `lora96`, `lora128` |
| `--checkpoint` | Single checkpoint file |
| `--checkpoint_root` | Directory of `.pt` checkpoints |
| `--lora_config` | Required for LoRA variants |
| `--output_dir` | Root output directory |
| `--gpu` | CUDA device id exposed through `CUDA_VISIBLE_DEVICES` |
| `--batch_size` | Inference batch size. Use `8` to reproduce previous LoRA runs when memory allows |
| `--resize` | Long-edge resize. `0` means original resolution |
| `--intrinsics_mode` | `auto`, `load`, or `none` |
| `--mask_mode` | Reserved. `none` and `load` are accepted, but mask evaluation is not implemented yet |
| `--cleanup_intermediate` | Delete large intermediate predictions after reports are generated |

Intrinsics modes:

| Mode | Behavior |
|---|---|
| `auto` | Use `meta.json` intrinsics when available |
| `load` | Require `meta.json` intrinsics for every image |
| `none` | Do not load or pass intrinsics |

## LoRA Evaluation Example

Run LoRA-96 on Bench and Oblique:

```bash
CUDA_VISIBLE_DEVICES=7 conda run -n moge310 python \
  /home/szq/moge2/MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint /path/to/Moge2-Aerial.pt \
  --lora_config /path/to/config-lora-all.json \
  --bench_input /path/to/Bench-ori \
  --bench_gt /path/to/Bench \
  --bench_csv_dir /path/to/Bench-ori \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --output_dir /path/to/eval_outputs \
  --gpu 7 \
  --resize 0 \
  --batch_size 8 \
  --intrinsics_mode none \
  --mask_mode none \
  --cleanup_intermediate
```

Use `--model_type lora64` or `--model_type lora128` for other LoRA ranks. The LoRA alpha is set automatically to `2 * rank`.

## Full / Head / Neck Evaluation Example

For non-LoRA checkpoints, omit `--lora_config`:

```bash
CUDA_VISIBLE_DEVICES=7 conda run -n moge310 python \
  /home/szq/moge2/MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type full \
  --checkpoint /path/to/full_model.pt \
  --bench_input /path/to/Bench-ori \
  --bench_gt /path/to/Bench \
  --bench_csv_dir /path/to/Bench-ori \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --output_dir /path/to/eval_outputs \
  --gpu 7 \
  --resize 0 \
  --batch_size 8 \
  --intrinsics_mode none \
  --cleanup_intermediate
```

Change `--model_type` to `head` or `neck` for the corresponding checkpoint.

## Checkpoint Directory Evaluation

Evaluate multiple checkpoints from a directory:

```bash
CUDA_VISIBLE_DEVICES=7 conda run -n moge310 python \
  /home/szq/moge2/MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint_root /path/to/checkpoints \
  --lora_config /path/to/config-lora-all.json \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --output_dir /path/to/eval_outputs \
  --gpu 7 \
  --step_interval 1000 \
  --batch_size 8 \
  --intrinsics_mode none
```

Useful checkpoint filters:

| Option | Description |
|---|---|
| `--step_interval 1000` | Evaluate checkpoints whose step is divisible by 1000 |
| `--exact_steps 1000,2400` | Always include these steps |
| `--min_step 1000` | Minimum checkpoint step |
| `--max_step 10000` | Maximum checkpoint step |
| `--include_latest` | Include `latest*.pt` |
| `--exclude_named_checkpoints` | Ignore non-numeric checkpoint names |

## Notes For Reproducibility

Use the same settings when comparing with previous results:

```text
model_type
checkpoint
batch_size
resize
intrinsics_mode
input dataset root
GT dataset root
Bench CSV directory
evaluation script version
```

`batch_size` can affect predictions because images in the same batch are padded to the maximum height and width in that batch. To reproduce old LoRA runs, use `--batch_size 8` when GPU memory allows.

Mask loading is intentionally exposed as `--mask_mode` for future compatibility, but it is currently not used by inference or evaluation.
