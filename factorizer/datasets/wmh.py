import sys

import numpy as np
import torch
from monai import transforms, data

from ..data import DataModule, ReadImaged, Renamed, Inferer


###################################
# Transform
###################################


def wmh_train_transform(
    spacing=(1.0, 1.0, 1.0), spatial_size=(128, 128, 128), num_patches=1
):
    train_transform = [
        ReadImaged(["image", "label"]),
        transforms.Lambdad("label", lambda x: (x == 1).astype(np.float32)),
        transforms.AddChanneld("label"),
        transforms.CropForegroundd(["image", "label"], source_key="image"),
        transforms.NormalizeIntensityd("image", channel_wise=True),
        transforms.Spacingd(
            ["image", "label"], pixdim=spacing, mode=("bilinear", "nearest"),
        ),
        transforms.SpatialPadd(["image", "label"], spatial_size=spatial_size),
        transforms.RandCropByPosNegLabeld(
            ["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1,
            neg=1,
            num_samples=num_patches,
            image_key="image",
            image_threshold=0,
        ),
        transforms.RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=spatial_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "nearest"),
            as_tensor_output=False,
        ),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandGaussianNoised("image", prob=0.15, std=0.1),
        transforms.RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        transforms.RandScaleIntensityd("image", prob=0.15, factors=0.3),
        transforms.RandShiftIntensityd("image", prob=0.15, offsets=0.1),
        transforms.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
        transforms.ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform


def wmh_val_transform():
    val_transform = [
        ReadImaged(["image", "label"], allow_missing_keys=True),
        transforms.Lambdad(
            "label",
            lambda x: (x == 1).astype(np.float32),
            allow_missing_keys=True,
        ),
        transforms.AddChanneld("label", allow_missing_keys=True),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform


def wmh_test_transform():
    return wmh_val_transform()


def wmh_vis_transform(spacing=(1.0, 1.0, 1.0)):
    vis_transform = [
        ReadImaged(["image", "label"], allow_missing_keys=True),
        transforms.Lambdad(
            "label",
            lambda x: (x == 1).astype(np.float32),
            allow_missing_keys=True,
        ),
        transforms.AddChanneld("label", allow_missing_keys=True),
        transforms.NormalizeIntensityd("image", channel_wise=True),
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    vis_transform = transforms.Compose(vis_transform)
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
        num_workers=None,
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
        dataset_cls = (data.CacheDataset, dataset_cls_params)

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
            post = transforms.Lambdad("input", lambda x: x)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "prob":
            post = transforms.Lambdad("input", torch.sigmoid)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "class":
            post = transforms.Lambdad("input", lambda x: x >= 0)
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

