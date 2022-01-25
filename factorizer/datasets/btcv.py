import sys

import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    ScaleIntensityRanged,
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
    AsDiscreted,
    ToTensord,
    Lambdad,
)
from monai.data import CacheDataset, SmartCacheDataset

from ..data import DataModule, Renamed, Inferer


###################################
# Transform
###################################


def btcv_train_transform(
    spacing=(1.0, 1.0, 1.0), spatial_size=(96, 96, 96), num_patches=1
):
    transforms = [
        LoadImaged(["image", "label"]),
        AddChanneld(["image", "label"]),
        AsDiscreted("label", to_onehot=14),
        CropForegroundd(["image", "label"], source_key="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
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
            mode=("bilinear", "bilinear"),
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
        AsDiscreted("label", argmax=True, to_onehot=14),
        ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = Compose(transforms)
    return train_transform


def btcv_val_transform():
    transforms = [
        LoadImaged(["image", "label"], allow_missing_keys=True),
        AddChanneld(["image", "label"]),
        AsDiscreted("label", to_onehot=14, allow_missing_keys=True),
        ScaleIntensityRanged(
            keys="image",
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = Compose(transforms)
    return val_transform


def btcv_test_transform():
    return btcv_val_transform()


def btcv_vis_transform(spacing=(1.0, 1.0, 1.0)):
    transforms = [
        LoadImaged(["image", "label"]),
        AddChanneld(["image", "label"]),
        AsDiscreted("label", to_onehot=14),
        ScaleIntensityRanged(
            keys="image",
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        AsDiscreted("label", argmax=True, to_onehot=14),
        ToTensord(["image", "label"]),
        Renamed(),
    ]
    vis_transform = Compose(transforms)
    return vis_transform


###################################
# Dataset
###################################


class BTCVDataModule(DataModule):
    def __init__(
        self,
        data_properties,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(96, 96, 96),
        num_patches=1,
        num_splits=5,
        split=0,
        batch_size=2,
        num_workers=None,
        num_init_workers=None,
        num_replace_workers=None,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        replace_rate=0.2,
        progress=True,
        shuffle=True,
        copy_cache=True,
        seed=42,
    ):
        train_dataset_params = {
            "cache_num": cache_num,
            "cache_rate": cache_rate,
            "replace_rate": replace_rate,
            "num_init_workers": num_init_workers,
            "num_replace_workers": num_replace_workers,
            "progress": progress,
            "shuffle": shuffle,
            "seed": seed,
            "copy_cache": copy_cache,
        }
        train_dataset_cls = (SmartCacheDataset, train_dataset_params)

        val_dataset_params = {
            "cache_num": cache_num,
            "cache_rate": cache_rate,
            "num_workers": num_workers,
            "progress": progress,
            "copy_cache": copy_cache,
        }
        val_dataset_cls = (CacheDataset, val_dataset_params)

        train_transform = btcv_train_transform(
            spacing, spatial_size, num_patches
        )
        val_transform = btcv_val_transform()
        test_transform = btcv_test_transform()
        vis_transform = btcv_vis_transform()
        super().__init__(
            data_properties,
            train_dataset_cls=train_dataset_cls,
            val_dataset_cls=val_dataset_cls,
            test_dataset_cls=val_dataset_cls,
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
BTCV = BTCVDataModule


###################################
# Inference
###################################


class BTCVInferer(Inferer):
    def __init__(
        self,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(96, 96, 96),
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
            post = Lambdad("input", lambda x: x.softmax(dim=1))
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "class":
            post = Lambdad("input", lambda x: x.argmax(dim=1))
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "one-hot":

            def get_onehot(x):
                y = x.argmax(dim=1, keepdim=True)
                result = [y == j for j in range(x.shape[1])]
                result = torch.cat(result, dim=1).float()
                return result

            post = Lambdad("input", get_onehot)
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

