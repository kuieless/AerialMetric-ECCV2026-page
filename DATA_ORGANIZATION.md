# Data Organization

This file defines the dataset layout expected by the AerialMetric aerial
inference and evaluation scripts. Paths below are examples; any root directory
is valid as long as the internal structure is unchanged.

## Minimal Inference Input

For inference only, place images under scene directories:

```text
input_root/
└── SceneName/
    └── image/
        ├── sample_0001.jpg
        └── sample_0002.png
```

The image folder may also be named `images`, `rgb`, `rgbs`, `color`, or `jpg`.
The output is written as one directory per image:

```text
output_root/
└── SceneName/
    └── sample_0001/
        └── depth.npy
```

If camera intrinsics are unavailable, run with `--intrinsics_mode none`. If
intrinsics are required, each sample can provide a `meta.json` file and the
command can use `--intrinsics_mode auto` or `--intrinsics_mode load`.

## Aerial Benchmark Inputs

The unified aerial evaluation command accepts separate roots for the three
evaluation splits:

```text
--bench_input /path/to/Bench-ori
--bench_gt /path/to/Bench
--bench_csv_dir /path/to/Bench-ori
--oblique_input /path/to/Oblique
--oblique_gt /path/to/Oblique
--wild_input /path/to/Wild
--wild_gt /path/to/Wild
```

### Bench Images

```text
Bench-ori/
├── Cleaned_Dataset_Campus/
│   └── image/
│       └── SampleID.jpg
├── Cleaned_Dataset_Factory/
│   └── image/
├── Cleaned_Dataset_Farm/
│   └── image/
└── Cleaned_Dataset_Gress/
    └── image/
```

Bench metadata CSV files are read from `--bench_csv_dir` when present:

```text
final_dataset_campus.csv
final_dataset_factory.csv
final_dataset_farm.csv
final_dataset_grass.csv
```

### Bench Ground Truth

```text
Bench/
└── SceneName/
    └── SampleID/
        ├── image.jpg
        ├── depth.npy
        └── meta.json
```

### Oblique And Wild Images

```text
Oblique/
└── SceneName/
    └── image/
        └── SampleID.jpg

Wild/
└── SceneName/
    └── image/
        └── SampleID.jpg
```

`rgbs/` is also accepted instead of `image/`.

### Oblique And Wild Ground Truth

```text
Oblique/
└── SceneName/
    ├── image/
    │   └── SampleID.jpg
    └── depth/
        └── SampleID.npy

Wild/
└── SceneName/
    ├── image/
    │   └── SampleID.jpg
    └── depth/
        └── SampleID.npy
```

Image stems and depth stems must match. For example,
`image/frame_0001.jpg` must correspond to `depth/frame_0001.npy`.

## Example Evaluation Command

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n moge310 python \
  MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint /path/to/Moge2-Aerial.pt \
  --lora_config MoGe/configs/Final_train/config-lora-all.json \
  --bench_input /path/to/Bench-ori \
  --bench_gt /path/to/Bench \
  --bench_csv_dir /path/to/Bench-ori \
  --oblique_input /path/to/Oblique \
  --oblique_gt /path/to/Oblique \
  --wild_input /path/to/Wild \
  --wild_gt /path/to/Wild \
  --output_dir /path/to/eval_outputs \
  --gpu 0 \
  --resize 0 \
  --batch_size 1 \
  --sampling_ratio 0.1 \
  --intrinsics_mode none \
  --mask_mode none
```

The evaluation writes reports under:

```text
eval_outputs/
└── model_type/
    └── checkpoint_name/
        └── Extracted/
            ├── Bench/Eval_Report_Bench.txt
            ├── Oblique/Eval_Report_Oblique_Pixel.txt
            └── Wild/Eval_Report_Wild_MultiRange.txt
```

## Common Checks

`Found 0 scenes to process` usually means the input root does not contain scene
directories with a supported image folder.

`No valid samples found` usually means image and depth stems do not match or GT
depth files are missing.

`No CUDA GPUs are available` means PyTorch cannot see the selected GPU. Check
the active conda environment, `CUDA_VISIBLE_DEVICES`, and `nvidia-smi`.
