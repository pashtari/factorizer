
# SegResNet for Ischemic Stroke Lesion Segmentation

This repository provides a MONAI bundle for SegResNet [1] to perform 3D volumetric segmentation of ischemic stroke lesions from multi-modal MRI scans. It uses data from the [ISLES 2022 Challenge](https://isles-challenge.org/) on ischemic stroke lesion segmentation.

## Overview

This model aims to automatically segment ischemic stroke lesions from multi-modal MRI data, with the modalities including DWI, ADC, and FLAIR. By default, this bundle uses only DWI and ADC, which are spatially aligned:

### Input Channels: 2
- **0**: DWI
- **1**: ADC

### Output Channels: 1
- **0**: Ischemic Stroke Lesion

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

The ISLES'22 dataset can be downloaded from the [official challenge website](https://isles-challenge.org/). Follow the challenge guidelines to download the training set.

Alternatively, you can automate the process and use [`data.sh`](data.sh) to download and unpack the dataset:

```bash
bash data.sh --data_dir <data_dir>
```

This command will create the required data folder `ISLES22` within the specified `<data_dir>`.

### 2. Generate Datalist

Use [`scripts/make_datalist.py`](scripts/make_datalist.py) to generate the necessary JSON datalist for training:

```bash
python scripts/make_datalist.py --data_dir /path/to/data
```

This creates the JSON datalist [`configs/datalist.json`](configs/datalist.json) needed for training, with the data partitioned using internal stratified 5-fold cross-validation.

## Training

The training configuration follows the methodology described in the Deconver paper [2]. Key settings include:

- **GPU Requirement**: At least 32GB (16GB) for a batch size of 2 (1)
- **Model Input Size**: 128 x 128 x 128
- **Optimizer**: AdamW
- **Initial Learning Rate**: 1e-4
- **Loss Function**: DiceCELoss

### Single-GPU Training
Use [`train.sh`](train.sh) to train on the first data fold using a single GPU:

```bash
bash train.sh --data_dir /path/to/data --fold 0 --batch_size 2
```

### Multi-GPU Training
Use [`train_multigpu.sh`](train_multigpu.sh) to train using multiple GPUs:

```bash
bash train_multigpu.sh --data_dir /path/to/data --fold 0 --batch_size 2
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
sbatch --job-name=segresnet_isles22_fold0 train.sh --conda_path /path/to/conda --env <env_name> --data_dir /path/to/data --fold 0 --batch_size 2
```
Adjust the SLURM configurations based on your cluster settings.

## Performance

The model achieves a Dice score of 75% in 5-fold cross-validation.

## Disclaimer

This software is provided for research purposes only. It is not intended for clinical or diagnostic use.

## References

[1] Myronenko, A. (2018). 3D MRI brain tumor segmentation using autoencoder regularization. *International MICCAI Brainlesion Workshop*. Springer, Cham. [https://doi.org/10.1007/978-3-030-11726-9_28](https://doi.org/10.1007/978-3-030-11726-9_28)

[2] Ashtari, P., et al. (2023). Deconver: Under Preparation.