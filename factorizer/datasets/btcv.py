import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from monai.apps import CrossValidation
from monai.data import DataLoader, list_data_collate
from monai.transforms import (
    Transform,
    MapTransform,
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    ScaleIntensityRanged,
    Spacingd,
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
from monai.inferers import SlidingWindowInferer

from .utils import StandardDataset, SaveImaged


###################################
# Transform
###################################


class OneHotEncoderd(MapTransform):
    """Convert labels to multi channels using one-hot encoder"""

    def __init__(self, keys, num_classes, allow_missing_keys=False):
        self.num_classes = num_classes
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            result = [d[key] == j for j in range(self.num_classes)]
            d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


class Binarized(MapTransform):
    """Discretize a logit or probability map (max -> 0, non-max -> 1)."""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        data = dict(data)
        for key in self.key_iterator(data):
            num_classes = data[key].shape[0]
            data[key] = np.argmax(data[key], axis=0)
            result = [data[key] == j for j in range(num_classes)]
            data[key] = np.stack(result, axis=0).astype(np.float32)

        return data


class Renamed(Transform):
    def __call__(self, data):
        if "image" in data:
            data["input"] = data.pop("image")

        if "image_transforms" in data:
            data["input_transforms"] = data.pop("image_transforms")

        if "image_meta_dict" in data:
            data["input_meta_dict"] = data.pop("image_meta_dict")

        if "label" in data:
            data["target"] = data.pop("label")

        if "label_transforms" in data:
            data["target_transforms"] = data.pop("label_transforms")

        if "label_meta_dict" in data:
            data["target_meta_dict"] = data.pop("label_meta_dict")

        data["id"] = os.path.basename(
            data["input_meta_dict"]["filename_or_obj"]
        ).split(".")[0]

        return data


def get_train_transform(spacing=(1.0, 1.0, 1.0), spatial_size=(128, 128, 128)):
    transforms = [
        LoadImaged(["image", "label"]),
        AddChanneld("image"),
        OneHotEncoderd("label", num_classes=14),
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
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1,
            neg=1,
            num_samples=2,
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
        Binarized("label"),
        ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = Compose(transforms)
    return train_transform


def get_val_transform():
    transforms = [
        LoadImaged(["image", "label"], allow_missing_keys=True),
        AddChanneld(["image"]),
        OneHotEncoderd("label", num_classes=14, allow_missing_keys=True),
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


def get_test_transform():
    return get_val_transform()


def get_vis_transform(spacing=(1.0, 1.0, 1.0)):
    transforms = [
        LoadImaged(["image", "label"]),
        AddChanneld(["image"]),
        OneHotEncoderd("label", num_classes=14),
        ScaleIntensityRanged(
            keys="image",
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Spacingd(
            keys=["input", "target"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        Binarized("label"),
        ToTensord(["image", "label"]),
        Renamed(),
    ]
    vis_transform = Compose(transforms)
    return vis_transform


###################################
# Dataset
###################################


class BTCVDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
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
        super().__init__()
        self.root_dir = root_dir
        self.spacing = spacing
        self.spatial_size = spatial_size
        self.num_splits = num_splits
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.progress = progress
        self.copy_cache = copy_cache
        self.seed = seed

        # get training transform
        self.train_transform = get_train_transform(spacing, spatial_size)
        # get validation transform
        self.val_transform = get_val_transform()
        # get test transform
        self.test_transform = get_test_transform()
        # get visualization transform
        self.vis_transform = get_vis_transform()

        self.train_set = self.val_set = self.test_set = None

    def setup(self, stage):
        if stage in ("fit", "validate", None):
            # make training set
            train_cv = CrossValidation(
                dataset_cls=StandardDataset,
                nfolds=self.num_splits,
                seed=self.seed,
                root_dir=self.root_dir,
                section="training",
                transform=self.train_transform,
                cache_num=self.cache_num,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                progress=self.progress,
                copy_cache=self.copy_cache,
            )
            train_folds = [
                k for k in range(self.num_splits) if k != self.split
            ]
            self.train_set = train_cv.get_dataset(train_folds)

            # make validation set
            val_cv = CrossValidation(
                dataset_cls=StandardDataset,
                nfolds=self.num_splits,
                seed=self.seed,
                root_dir=self.root_dir,
                section="validation",
                transform=self.val_transform,
                cache_num=self.cache_num,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                progress=self.progress,
                copy_cache=self.copy_cache,
            )
            self.val_set = val_cv.get_dataset([self.split])

        if stage in ("test", "predict", None):
            # make test set
            self.test_set = StandardDataset(
                self.root_dir,
                section="test",
                transform=self.test_transform,
                cache_num=self.cache_num,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                progress=self.progress,
                copy_cache=self.copy_cache,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set, batch_size=1, num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set, batch_size=1, num_workers=self.num_workers,
        )
        return test_loader


# alias
BTCV = BTCVDataModule

###################################
# Inference
###################################


class Interpolate(object):
    def __init__(self, key, spacing, **kwargs):
        super().__init__()
        self.key = key
        self.meta_key = f"{key}_meta_dict"
        self.spacing = torch.tensor(spacing)
        self.orig_spacing = None
        self.orig_size = None
        self.new_size = None
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, batch):
        out = copy.deepcopy(batch)
        self.orig_spacing = batch[self.meta_key]["pixdim"][0][1:4].to("cpu")
        self.orig_size = batch[self.meta_key]["dim"][0][1:4].to("cpu")
        self.new_size = self.orig_size * self.orig_spacing / self.spacing
        self.new_size = self.new_size.round().to(device="cpu", dtype=torch.int)
        out[self.key] = F.interpolate(
            batch[self.key], size=tuple(self.new_size.numpy()), **self.kwargs
        )
        return out

    def inverse_transform(self, batch):
        out = copy.deepcopy(batch)
        out[self.key] = F.interpolate(
            batch[self.key], size=tuple(self.orig_size.numpy()), **self.kwargs
        )
        return out


class Inferer(object):
    def __init__(
        self,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
        post="logit",
        write_dir=None,
        output_dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.device = None

        # downsampling
        self.interp = Interpolate("input", spacing=spacing, mode="trilinear")

        # inference method
        self.inferer = SlidingWindowInferer(roi_size=spatial_size, **kwargs)

        # postprocessing transforms
        if post == "logit":
            self.post = Lambdad("input", lambda x: x)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "prob":
            self.post = Lambdad("input", lambda x: x.softmax(dim=1))
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "class":
            self.post = Lambdad("input", lambda x: x.argmax(dim=1))
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "one-hot":

            def get_onehot(x):
                y = x.argmax(dim=1, keepdim=True)
                result = [y == j for j in range(x.shape[1])]
                result = torch.cat(result, dim=1).float()
                return result

            self.post = Lambdad("input", get_onehot)
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        else:
            self.post = post

        # write to file
        self.output_dtype = output_dtype
        self.write_dir = write_dir

    def get_preprocessed(self, batch, model):
        batch = self.interp.transform(batch)
        return batch

    def get_inferred(self, batch, model):
        batch = self.get_preprocessed(batch, model)
        batch["input"] = self.inferer(batch["input"], model)
        return batch

    def get_postprocessed(self, batch, model):
        batch = self.get_inferred(batch, model)
        batch = self.interp.inverse_transform(batch)
        batch = self.post(batch)
        return batch

    def write(self, batch, write_dir):
        if write_dir is not None:
            save = SaveImaged(
                "input",
                output_dir=write_dir,
                output_dtype=self.output_dtype,
                output_postfix="",
                save_batch=True,
                print_log=True,
            )
            save(batch)

    def __call__(self, batch, model):
        batch = self.get_postprocessed(batch, model)
        self.write(batch, self.write_dir)
        return batch["input"]

