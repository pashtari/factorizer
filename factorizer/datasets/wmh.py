import sys

import numpy as np
import torch
from monai.transforms import (
    Compose,
    AddChanneld,
    CropForegroundd,
    NormalizeIntensityd,
    Spacingd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ToTensord,
    Lambdad,
)
from monai.data import CacheDataset

from ..data import DataModule, ReadImaged, Renamed, Inferer


###################################
# Transform
###################################


def wmh_train_transform(
    spacing=(1.0, 1.0, 1.0), spatial_size=(128, 128, 128), num_patches=1
):
    transforms = [
        ReadImaged(["image", "label"]),
        Lambdad("label", lambda x: (x == 1).astype(np.float32)),
        AddChanneld("label"),
        CropForegroundd(["image", "label"], source_key="image"),
        NormalizeIntensityd("image", channel_wise=True),
        Spacingd(
            ["image", "label"], pixdim=spacing, mode=("bilinear", "nearest"),
        ),
        SpatialPadd(["image", "label"], spatial_size=spatial_size),
        RandCropByPosNegLabeld(
            ["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1,
            neg=1,
            num_samples=num_patches,
            image_key="image",
            image_threshold=0,
        ),
        RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=spatial_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "nearest"),
            as_tensor_output=False,
        ),
        RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        RandGaussianNoised("image", prob=0.15, std=0.1),
        RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        RandScaleIntensityd("image", prob=0.15, factors=0.3),
        RandShiftIntensityd("image", prob=0.15, offsets=0.1),
        RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
        ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = Compose(transforms)
    return train_transform


def wmh_val_transform():
    transforms = [
        ReadImaged(["image", "label"], allow_missing_keys=True),
        Lambdad(
            "label",
            lambda x: (x == 1).astype(np.float32),
            allow_missing_keys=True,
        ),
        AddChanneld("label", allow_missing_keys=True),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = Compose(transforms)
    return val_transform


def wmh_test_transform():
    return wmh_val_transform()


def wmh_vis_transform(spacing=(1.0, 1.0, 1.0)):
    transforms = [
        ReadImaged(["image", "label"], allow_missing_keys=True),
        Lambdad(
            "label",
            lambda x: (x == 1).astype(np.float32),
            allow_missing_keys=True,
        ),
        AddChanneld("label", allow_missing_keys=True),
        NormalizeIntensityd("image", channel_wise=True),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    vis_transform = Compose(transforms)
    return vis_transform


###################################
# Data module
###################################


class WMHDataModule(DataModule):
    def __init__(
        self,
        data_properties,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
        num_patches=1,
        num_splits=5,
        split=0,
        batch_size=2,
        num_workers=0,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        progress=True,
        copy_cache=True,
        seed=42,
    ):
        dataset_cls_params = {
            "cache_num": cache_num,
            "cache_rate": cache_rate,
            "num_workers": num_workers,
            "progress": progress,
            "copy_cache": copy_cache,
        }
        dataset_cls = (CacheDataset, dataset_cls_params)

        train_transform = wmh_train_transform(
            spacing, spatial_size, num_patches
        )
        val_transform = wmh_val_transform()
        test_transform = wmh_test_transform()
        vis_transform = wmh_vis_transform(spacing)
        super().__init__(
            data_properties,
            train_dataset_cls=dataset_cls,
            val_dataset_cls=dataset_cls,
            test_dataset_cls=dataset_cls,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            vis_transform=vis_transform,
            num_splits=num_splits,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )


# alias
WMH = WMHDataModule


###################################
# Inference
###################################


class WMHInferer(Inferer):
    def __init__(
        self,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
        post=None,
        write_dir=None,
        output_dtype=None,
        **kwargs,
    ) -> None:

        # postprocessing transforms
        if post == "logit":
            post = Lambdad("input", lambda x: x)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "prob":
            post = Lambdad("input", torch.sigmoid)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "class":
            post = Lambdad("input", lambda x: x >= 0)
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        else:
            post = post

        super().__init__(
            spacing=spacing,
            spatial_size=spatial_size,
            post=post,
            write_dir=write_dir,
            output_dtype=output_dtype,
            **kwargs,
        )

