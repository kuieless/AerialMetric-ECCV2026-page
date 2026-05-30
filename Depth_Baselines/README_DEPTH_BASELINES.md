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

These baselines should be prepared according to their official repositories:

| Model | Official repo | Environment setup |
|---|---|---|
| DepthPro | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) | `conda create -n depth-pro -y python=3.9`<br>`conda activate depth-pro`<br>`pip install -e .`<br>`source get_pretrained_models.sh` |
| ZoeDepth | [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth) | `mamba env create -n zoe --file environment.yml`<br>`mamba activate zoe`<br>or `conda env create -n zoe --file environment.yml`<br>`conda activate zoe` |
| UniDepth | [lpiccinelli-eth/unidepth](https://github.com/lpiccinelli-eth/unidepth) | `python -m venv <YOUR-VENVS-DIR>/Unidepth`<br>`source <YOUR-VENVS-DIR>/Unidepth/bin/activate`<br>`pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118`<br>`cd unidepth/ops/knn; bash compile.sh; cd ../../../` |

Official setup notes:

- DepthPro recommends Python 3.9 and a local editable install from the repo root.
- ZoeDepth uses the repository-provided `environment.yml`.
- UniDepth requires Linux, Python 3.10+, CUDA 11.8+, and the KNN extension compiled for evaluation.

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
