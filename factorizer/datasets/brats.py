import sys

import numpy as np
import torch
from monai import transforms, data

from ..data import DataModule, Renamed, Inferer


###################################
# Transform
###################################


class BraTSOneHotEncoderd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 0: background, 
    label 1: edema (ED) 
    label 2: necrotic and non-enhancing tumor (NCR&NET) 
    label 3: enhancing tumor (ET)
    
    If nested = True, the classes are enhancing tumor (ET), tumor core (TC), 
    and whole tumor (WT):

    ET: label 3
    TC: NCR&NET (label 2) + ET (label 3)
    WT: ET (label 3) + ED (label 1) + NCR&NET (label 2) 

    """

    def __init__(self, keys, nested=True, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.nested = nested

    def __call__(self, data):
        if self.nested:
            d = dict(data)
            for key in self.key_iterator(d):
                result = []
                # ET (label 3)
                result.append(d[key] == 3)
                # TC: ET (label 3) + NCR&NET (label 2)
                result.append(np.logical_or(d[key] == 3, d[key] == 2))
                # WT: ET (label 3) + NCR&NET (label 2) + ED (label 1)
                result.append(
                    np.logical_or(
                        np.logical_or(d[key] == 3, d[key] == 2), d[key] == 1
                    )
                )
                d[key] = np.stack(result, axis=0).astype(np.float32)
        else:
            d = dict(data)
            for key in self.key_iterator(d):
                result = []
                # ED (label 1)
                result.append(d[key] == 1)
                # NCR&NET (label 2)
                result.append(d[key] == 2)
                # ET (label 3)
                result.append(d[key] == 3)
                d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


def brats_train_transform(spatial_size=(128, 128, 128)):
    train_transform = [
        transforms.LoadImaged(["image", "label"]),
        transforms.AsChannelFirstd("image"),
        BraTSOneHotEncoderd("label", nested=True),
        transforms.CropForegroundd(["image", "label"], source_key="image"),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.RandSpatialCropd(
            ["image", "label"], roi_size=spatial_size, random_size=False
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
        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform


def brats_val_transform():
    val_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.AsChannelFirstd("image"),
        BraTSOneHotEncoderd("label", nested=True, allow_missing_keys=True),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform


def brats_test_transform():
    return brats_val_transform()


def brats_vis_transform():
    vis_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.AsChannelFirstd("image"),
        BraTSOneHotEncoderd("label", nested=False, allow_missing_keys=True),
        transforms.NormalizeIntensityd("image", channel_wise=True),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    vis_transform = transforms.Compose(vis_transform)
    return vis_transform


###################################
# Data module
###################################


class BraTSDataModule(DataModule):
    def __init__(
        self,
        data_properties,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
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

        train_transform = brats_train_transform(spatial_size)
        val_transform = brats_val_transform()
        test_transform = brats_test_transform()
        vis_transform = brats_vis_transform()
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
BRATS = BraTSDataModule


###################################
# Inference
###################################


class BraTSInferer(Inferer):
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
        if post == "logit-nested":
            post = transforms.Lambdad("input", lambda x: x)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "prob-nested":
            post = transforms.Lambdad("input", lambda x: x.sigmoid(dim=1))
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "one-hot-nested":
            post = transforms.Lambdad(
                "input", lambda x: torch.where(x >= 0, 1.0, 0.0)
            )
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "class":

            def func(x):
                mask = torch.where(x >= 0, 1.0, 0.0)
                et = mask[:, 0, ...]
                tc = mask[:, 1, ...]
                wt = mask[:, 2, ...]
                out = torch.zeros_like(wt)
                out[et == 1] = 3
                out[torch.logical_and(tc == 1, et == 0)] = 2
                out[torch.logical_and(wt == 1, tc == 0)] = 1
                return out

            self.post = transforms.Lambdad("input", func)
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "one-hot":

            def func(x):
                mask = torch.where(x >= 0, 1.0, 0.0)
                et = mask[:, 0:1, ...]
                tc = mask[:, 1:2, ...]
                wt = mask[:, 2:3, ...]
                bg = (wt != 0).float()
                nt = torch.logical_and(tc == 1, et == 0).float()
                ed = torch.logical_and(wt == 1, tc == 0).float()
                return torch.cat((bg, ed, nt, et), dim=1)

            post = transforms.Lambdad("input", func)
            output_dtype = np.uint8 if output_dtype is None else output_dtype

        super().__init__(
            spacing=spacing,
            spatial_size=spatial_size,
            post=post,
            write_dir=write_dir,
            output_dtype=output_dtype,
            **kwargs,
        )

