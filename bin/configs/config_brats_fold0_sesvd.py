import sys

from torch import nn, optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from monai.losses import DiceCELoss

import factorizer as ft
from factorizer import datasets
from factorizer.utils.losses import DeepSuprLoss
from factorizer.utils.metrics import DiceMetric, HausdorffDistanceMetric
from factorizer.utils.schedulers import WarmupCosineSchedule


CONFIG = {
    "data": {
        "datamodule": (
            datasets.BRATS,
            {
                "root_dir": "/data/leuven/336/vsc33647/data/Decathlon/Task01_BrainTumour",
                "spatial_size": (128, 128, 128),
                "num_splits": 5,
                "split": 0,
                "batch_size": 1,
                "num_workers": 4,
                "cache_num": 4,
                "cache_rate": 1.0,
                "seed": 42,
            },
        )
    },
    "model": {
        "network": (
            ft.SegmentationFactorizer,
            {
                "in_channels": 4,
                "out_channels": 3,
                "spatial_size": (128, 128, 128),
                "encoder_depth": (1, 1, 1, 1, 1),
                "encoder_width": (32, 64, 128, 256, 512),
                "strides": (1, 2, 2, 2, 2),
                "decoder_depth": (1, 1, 1, 1),
                "stem": (
                    nn.Conv3d,
                    {"kernel_size": 3, "padding": 1, "bias": False},
                ),
                "downsample": (nn.Conv3d, {"kernel_size": 2, "bias": False}),
                "upsample": (nn.ConvTranspose3d, {"kernel_size": 2}),
                "head": (nn.Conv3d, {"kernel_size": 1}),
                "num_deep_supr": 3,
                "dropout": 0.1,
                "svd": (
                    ft.FactorizerSubblock,
                    {
                        "tensorize": (
                            ft.Matricize,
                            {"head_dim": 32, "grid_size": 1},
                        ),
                        "act": nn.GELU,
                        "factorize": ft.SVD,
                        "compression": 10,
                        "no_grad": True,
                        "dropout": 0.1,
                    },
                ),
                "p2p": (
                    ft.FactorizerSubblock,
                    {
                        "tensorize": (
                            ft.Matricize,
                            {"head_dim": 1, "patch_size": 8},
                        ),
                        "act": nn.GELU,
                        "factorize": ft.DepthWiseP2P,
                        "dropout": 0.1,
                    },
                ),
                "mlp": (ft.MLP, {"ratio": 2, "dropout": 0.1}),
            },
        ),
        "inferer": datasets.brats.Inferer(
            spatial_size=(128, 128, 128), overlap=0.5, post="one-hot-nested",
        ),
    },
    "optimization": {
        "loss": DeepSuprLoss(
            DiceCELoss,
            include_background=True,
            sigmoid=True,
            squared_pred=True,
        ),
        "metrics": {
            "dice": DiceMetric(
                include_background=True, include_zero_masks=True
            ),
            "hd": HausdorffDistanceMetric(
                include_background=True, include_zero_masks=True, percentile=95
            ),
        },
        "optimizer": (optim.AdamW, {"lr": 1e-4, "weight_decay": 1e-2}),
        "scheduler": (
            WarmupCosineSchedule,
            {"warmup_steps": 2000, "total_steps": 100000},
        ),
        "scheduler_config": {"interval": "step"},
    },
    "training": {
        "max_steps": 100000,
        "max_epochs": sys.maxsize,
        "gpus": 2,
        "num_nodes": 1,
        "accelerator": "ddp",
        "check_val_every_n_epoch": 50,
        "callbacks": [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_train_steps=50),
        ],
        "logger": TensorBoardLogger(save_dir="logs/brats/fold0", name="sesvd"),
    },
    "test": {
        "checkpoint": {
            "checkpoint_path": "logs/brats/fold0/sesvd/version_0/checkpoints/epoch=515-step=99999.ckpt",
        },
        "inferer": datasets.brats.Inferer(
            spatial_size=(128, 128, 128),
            overlap=0.5,
            post="class",
            write_dir="logs/brats/fold0/sesvd/version_0/predictions",
        ),
        "save_path": "logs/brats/fold0/sesvd/version_0/results.csv",
    },
}
