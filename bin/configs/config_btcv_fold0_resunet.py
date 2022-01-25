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
            datasets.BTCV,
            {
                "data_properties": "/data/leuven/336/vsc33647/data/BTCV/dataset.json",
                "spacing": (1.5, 1.5, 2.0),
                "spatial_size": (96, 96, 96),
                "num_patches": 1,
                "num_splits": 5,
                "split": 0,
                "batch_size": 1,
                "num_workers": 2,
                "num_init_workers": 2,
                "num_replace_workers": 2,
                "cache_num": 5,
                "replace_rate": 0.2,
                "seed": 42,
            },
        )
    },
    "model": {
        "network": (
            ft.UNet,
            {
                "in_channels": 1,
                "out_channels": 14,
                "spatial_dims": 3,
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
                "block": ft.Same(
                    (
                        ft.BasicBlock,
                        {
                            "conv": (
                                nn.Conv3d,
                                {"kernel_size": 3, "padding": 1},
                            ),
                            "norm": (nn.GroupNorm, (8,)),
                            "act": nn.LeakyReLU,
                        },
                    )
                ),
            },
        ),
        "inferer": datasets.BTCVInferer(
            spacing=(1.5, 1.5, 2.0),
            spatial_size=(96, 96, 96),
            overlap=0.5,
            post="one-hot",
        ),
    },
    "optimization": {
        "loss": DeepSuprLoss(
            DiceCELoss,
            include_background=False,
            softmax=True,
            squared_pred=True,
        ),
        "metrics": {
            "dice": DiceMetric(
                include_background=False, include_zero_masks=False
            ),
            "hd": HausdorffDistanceMetric(
                include_background=False,
                include_zero_masks=False,
                percentile=95,
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
        "check_val_every_n_epoch": 3000,
        "callbacks": [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_train_steps=50),
            ft.data.SmartDatasetCallback(),
        ],
        "logger": TensorBoardLogger(
            save_dir="logs/btcv/fold0", name="resunet"
        ),
    },
    "test": {
        "checkpoint": {
            "checkpoint_path": "logs/btcv/fold0/resunet/version_0/checkpoints/epoch=257-step=99999.ckpt",
        },
        "inferer": datasets.BTCVInferer(
            spacing=(1.5, 1.5, 2.0),
            spatial_size=(96, 96, 96),
            overlap=0.5,
            post="class",
            write_dir="logs/btcv/fold0/resunet/version_0/predictions",
        ),
        "save_path": "logs/btcv/fold0/resunet/version_0/results.csv",
    },
}
