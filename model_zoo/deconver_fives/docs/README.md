# Deconver for Retinal Vessel Segmentation

This repository provides a MONAI bundle for Deconver [1] to perform retinal vessel segmentation from fundus images. It uses data from the [FIVES](https://www.synapse.org/Synapse:syn51156910/).

## Overview

This model takes an RGB fundus image as input and segments retinal vessels.

### Input Channels: 4
- **0**: Red
- **1**: Green
- **2**: Blue

### Output Channels: 3
- **0**: Retinal Vessels

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [SLURM Support](#slurm-support)
- [Performance](#performance)
- [Disclaimer](#disclaimer)
- [References](#references)

## Installation

Run [`setup.sh`](setup.sh) to setup your Python environment and install dependencies. Specify the device (CPU or CUDA) and Python version.

**CPU Setup**:
```bash
bash setup.sh --device cpu --env <env_name> --python_version 3.12
```

**CUDA Setup**:
```bash
bash setup.sh --device cuda --env <env_name> --python_version 3.12
```

## Data

### 1. Download Dataset

The FIVES dataset can be downloaded from [here](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169).

Alternatively, you can automate the process and use [`data.sh`](data.sh) to download and unpack the dataset:

```bash
bash data.sh --data_dir <data_dir>
```

### 2. Generate Datalist

Use [`scripts/make_datalist.py`](scripts/make_datalist.py) to generate the necessary JSON datalist for training:

```bash
python scripts/make_datalist.py --data_dir /path/to/data
```

This creates the JSON datalist [`configs/datalist.json`](configs/datalist.json) needed for training, with the data partitioned using internal stratified 5-fold cross-validation.

## Training

The training configuration follows the methodology described in the Deconver paper [1]. Key settings include:

- **GPU Requirement**: At least 32GB (16GB) for a batch size of 16 (8)
- **Model Input Size**: 512 x 512
- **Optimizer**: AdamW
- **Initial Learning Rate**: 1e-4
- **Loss Function**: DiceCELoss

### Single-GPU Training
Use [`train.sh`](train.sh) to train on the first data fold using a single GPU:

```bash
bash train.sh --data_dir /path/to/data --fold 0 --batch_size 16
```

### Multi-GPU Training
Use [`train_multigpu.sh`](train_multigpu.sh) to train using multiple GPUs:

```bash
bash train_multigpu.sh --data_dir /path/to/data --fold 0 --batch_size 16
```

## Evaluation

Use [`evaluate.sh`](evaluate.sh) to evaluate a pre-trained model checkpoint (`model/model_fold=0.pt`) on the first fold:

```bash
bash evaluate.sh --data_dir /path/to/data --fold 0 --ckpt_name "model_fold=0.pt"
```

## Inference

Use [`inference.sh`](inference.sh) to perform inference using pre-trained model checkpoints (`models/*.pt`) on the validation dataset:

```bash
bash inference.sh --data_dir /path/to/data
```

The predictions will be saved in `~/outputs` by default.


## SLURM Support

All training ([`train.sh`](train.sh), [`train_multigpu.sh`](train_multigpu.sh)), evaluation ([`evaluate.sh`](evaluate.sh)), and inference ([`inference.sh`](inference.sh)) scripts are SLURM-compatible for submission on HPC clusters. Each script contains example SLURM configurations.

**Example SLURM Submission for Single-GPU Training**:
```bash
sbatch --job-name=deconver_fives_fold0 train.sh --conda_path /path/to/conda --env <env_name> --data_dir /path/to/data --fold 0 --batch_size 16
```
Adjust the SLURM configurations based on your cluster settings.

## Performance

The model achieves a Dice score of 90% in 5-fold cross-validation.

## Disclaimer

This software is provided for research purposes only. It is not intended for clinical or diagnostic use.

## References

[1] Ashtari, P., et al. (2023). Deconver: Under Preparation.

