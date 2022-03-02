import sys

import numpy as np
import torch
from monai import transforms, data

from ..data import DataModule, Renamed, Inferer


###################################
# Transform
###################################


def lits_train_transform(
    spacing=(1.5, 1.5, 2.0), spatial_size=(96, 96, 96), num_patches=1
):
    train_transform = [
        transforms.LoadImaged(["image", "label"]),
        transforms.AddChanneld(["image", "label"]),
        transforms.AsDiscreted("label", to_onehot=3),
        transforms.CropForegroundd(["image", "label"], source_key="image"),
        transforms.ScaleIntensityRanged(
            "image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True,
        ),
        transforms.Spacingd(
            ["image", "label"], pixdim=spacing, mode=("bilinear", "bilinear"),
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
            mode=("bilinear", "bilinear"),
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
        transforms.AsDiscreted("label", argmax=True, to_onehot=3),
        transforms.ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform


def lits_val_transform():
    val_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.AddChanneld(["image", "label"]),
        transforms.AsDiscreted("label", to_onehot=3, allow_missing_keys=True),
        transforms.ScaleIntensityRanged(
            keys="image",
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform


def lits_test_transform():
    return lits_val_transform()


def lits_vis_transform(spacing=(1.5, 1.5, 2.0)):
    vis_transform = [
        transforms.LoadImaged(["image", "label"]),
        transforms.AddChanneld(["image", "label"]),
        transforms.AsDiscreted("label", to_onehot=3),
        transforms.ScaleIntensityRanged(
            "image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True,
        ),
        transforms.Spacingd(
            ["image", "label"], pixdim=spacing, mode=("bilinear", "bilinear"),
        ),
        transforms.AsDiscreted("label", argmax=True, to_onehot=3),
        transforms.ToTensord(["image", "label"]),
        Renamed(),
    ]
    vis_transform = transforms.Compose(vis_transform)
    return vis_transform


###################################
# Dataset
###################################


class LiTSDataModule(DataModule):
    def __init__(
        self,
        data_properties,
        spacing=(1.5, 1.5, 2.0),
        spatial_size=(96, 96, 96),
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

        train_transform = lits_train_transform(
            spacing, spatial_size, num_patches
        )
        val_transform = lits_val_transform()
        test_transform = lits_test_transform()
        vis_transform = lits_vis_transform()
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
LITS = LiTSDataModule

###################################
# Inference
###################################


class LiTSInferer(Inferer):
    def __init__(
        self,
        spacing=(1.5, 1.5, 2.0),
        spatial_size=(96, 96, 96),
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
            post = transforms.Lambdad("input", lambda x: x.softmax(dim=1))
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "class":
            post = transforms.Lambdad("input", lambda x: x.argmax(dim=1))
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "one-hot":

            def get_onehot(x):
                y = x.argmax(dim=1, keepdim=True)
                result = [y == j for j in range(x.shape[1])]
                result = torch.cat(result, dim=1).float()
                return result

            post = transforms.Lambdad("input", get_onehot)
            output_dtype = np.uint8 if output_dtype is None else output_dtype

        super().__init__(
            spacing=spacing,
            spatial_size=spatial_size,
            post=post,
            write_dir=write_dir,
            output_dtype=output_dtype,
            **kwargs,
        )

