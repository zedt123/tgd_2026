# Tempered Guided Diffusion (PyTorch)

This repository contains the PyTorch implementation and experiment scripts for the NeurIPS submission. The instructions below describe how to install the environment, prepare the datasets/checkpoints, and run the experiments reported in the paper and appendix.

## Environment setup

The code was developed with:

* Python 3.11
* PyTorch with CUDA support on Linux
* `uv` for dependency management

Create the environment from the locked dependencies:

```bash
uv sync --locked
```

Activate the environment:

```bash
source .venv/bin/activate
```

All commands below assume they are run from the repository root after activating the environment.

To verify the installed PyTorch build can see the GPU, run:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

## Repository structure

```text
configs/        Experiment configurations
scripts/        Experiment launchers and analysis scripts
src/            Source code for the method and utilities
data/           Dataset instructions and local dataset location
model_weights/  Pretrained diffusion model weights
results/        Default output directory
```

## Data

Dataset licensing and setup instructions are provided in:

```text
data/README.md
```

Please follow the instructions there before running the image experiments. In particular, ImageNet images are not redistributed in this repository and must be obtained separately by users who agree to the ImageNet terms and conditions.

## Model weights

The image experiments use the public FFHQ and ImageNet pixel-space diffusion checkpoints used by DAPS.

Create the expected weights directory:

```bash
mkdir -p model_weights
```

Download the public FFHQ and ImageNet pixel diffusion checkpoints from:

```text
https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh
```

The files must be placed at exactly:

```text
model_weights/ffhq_10m.pt
model_weights/imagenet256.pt
```

For example:

```bash
mv {DOWNLOAD_DIR}/ffhq_10m.pt model_weights/ffhq_10m.pt
mv {DOWNLOAD_DIR}/imagenet256.pt model_weights/imagenet256.pt
```

## Running experiments

All experiment outputs are saved under:

```text
results/
```

## Main image experiments

To reproduce the main image experiment tables, run one sweep per dataset/task pair:

```bash
python -m scripts.sweep --config-name image_configs/<dataset>_<task>/table_results
```

where `<dataset>` is one of:

```text
ffhq
imagenet
```

and `<task>` is one of:

```text
ip
pr
```

For example:

```bash
python -m scripts.sweep --config-name image_configs/ffhq_pr/table_results
```

## Figure 2: 2D toy experiments

To reproduce the 2D toy experiments shown in Figure 2, run:

```bash
python -m scripts.toy2d_sweeps
```

## Figure 3: Time plots

To reproduce the time plots shown in Figure 3, run:

```bash
python -m scripts.ablation_time
```

## Appendix ablations

Appendix ablations are run with the same sweep entry point and the corresponding configuration files in `configs/image_configs/ablations`.

Template:

```bash
python -m scripts.sweep --config-name image_configs/ablations/<dataset>_<task>/<ABLATION>
```

Example:

```bash
python -m scripts.sweep --config-name image_configs/ablations/ffhq_pr/reconstruction
```

Please see the configuration files under `configs/` for the full list of ablation settings used in the appendix.

## Notes

* The repository uses `pyproject.toml` and `uv.lock`; no `requirements.txt` is needed.