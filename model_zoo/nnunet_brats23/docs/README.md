# nnU-Net for Brain Tumor Segmentation

This repository provides a MONAI bundle for nnU-Net to perform 3D volumetric segmentation of brain tumor subregions from multi-parametric MRI (mpMRI) scans. It uses data from the [BraTS 2023 - Adult Glioma Challenge](https://www.synapse.org/Synapse:syn51156910/). nnU-Net is detailed in [this paper](https://doi.org/10.1038/s41592-020-01008-z).

## Overview

This model takes an mpMRI scan consisting of four co-registered images (T1Gd, T1, T2, FLAIR) as input and segments three nested subregions of gliomas:

1. **Enhancing Tumor (ET)**: Highlights the active tumor regions, appearing hyperintense in T1Gd relative to T1.
2. **Tumor Core (TC)**: Represents the main tumor mass, which is typically resected. It encompasses enhancing, necrotic, and non-enhancing solid components.
3. **Whole Tumor (WT)**: Encompasses the entire tumor mass, including the tumor core and peritumoral edema, which is often hyperintense in FLAIR.

### Input Channels: 4
- **0**: FLAIR
- **1**: T1
- **2**: T1Gd (contrast-enhanced T1)
- **3**: T2

### Output Channels: 3
- **0**: Enhancing Tumor (ET)
- **1**: Tumor Core (TC)
- **2**: Whole Tumor (WT)

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

The dataset is available from the [BraTS 2023 - Adult Glioma Challenge](https://www.synapse.org/Synapse:syn51156910/). To download the training and validation datasets, you can follow the procedure described on the [Synapse website](https://www.synapse.org/Synapse:syn51156910/wiki/627000).

Alternatively, ensure that you have valid Synapse credentials and use [`data.sh`](data.sh) to automatically download and unzip the data files, 

```bash
bash data.sh --data_dir /path/to/data
```

This will create data directories named `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` and `ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData` in `/path/to/data`.

### 2. Generate Datalist

Use [`scripts/make_datalist.py`](scripts/make_datalist.py) to generate the necessary JSON datalist for training:

```bash
python scripts/make_datalist.py --data_dir /path/to/data
```

This creates the JSON datalist [`configs/datalist.json`](configs/datalist.json) needed for training, with the data partitioned using internal stratified 5-fold cross-validation.

## Training

The training configuration follows the methodology described in the nnU-Net paper [1]. Key settings include:

- **GPU Requirement**: At least 32GB (16GB) for a batch size of 2 (1)
- **Model Input Size**: 128 x 128 x 128
- **Optimizer**: AdamW
- **Initial Learning Rate**: 4e-4
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
sbatch --job-name=nnunet_brats23_fold0 train.sh --conda_path /path/to/conda --env <env_name> --data_dir /path/to/data --fold 0 --batch_size 2
```
Adjust the SLURM configurations based on your cluster settings.

## Performance

The model achieved the following Dice scores (%) in 5-fold cross-validation:

- **Enhancing Tumor (ET)**: 79.33%
- **Tumor Core (TC)**: 83.14%
- **Whole Tumor (WT)**: 90.16%
- **Average Score**: 84.21%

## Disclaimer

This software is provided for research purposes only. It is not intended for clinical or diagnostic use.

## References

[1] Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211. [https://doi.org/10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

