import os
import sys

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from monai.apps import CrossValidation
from monai.data import DataLoader
from monai.transforms import (
    Transform,
    MapTransform,
    Compose,
    LoadImaged,
    AsChannelFirstd,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    # RandBiasFieldd,
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


class Binarized(MapTransform):
    """Discretize a probability map with a threshold."""

    def __init__(self, keys, threshold=0.5, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data):
        data = dict(data)
        for key in self.key_iterator(data):
            data[key] = np.where(data[key] >= self.threshold, 1.0, 0.0)

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


def get_train_transform(spatial_size=(128, 128, 128)):
    transforms = [
        LoadImaged(["image", "label"]),
        AsChannelFirstd("image"),
        OneHotEncoderd("label", nested=True),
        CropForegroundd(["image", "label"], source_key="image"),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        RandSpatialCropd(
            ["image", "label"], roi_size=spatial_size, random_size=False
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
        # RandBiasFieldd("image", prob=0.15, degree=3, coeff_range=(0.0, 0.1)),
        RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
        Binarized("label", threshold=0.5),
        ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = Compose(transforms)
    return train_transform


def get_val_transform():
    transforms = [
        LoadImaged(["image", "label"], allow_missing_keys=True),
        AsChannelFirstd("image"),
        OneHotEncoderd("label", nested=True, allow_missing_keys=True),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = Compose(transforms)
    return val_transform


def get_test_transform():
    return get_val_transform()


def get_vis_transform():
    transforms = [
        LoadImaged(["image", "label"], allow_missing_keys=True),
        AsChannelFirstd("image"),
        OneHotEncoderd("label", nested=False, allow_missing_keys=True),
        NormalizeIntensityd("image", channel_wise=True),
        ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    vis_transform = Compose(transforms)
    return vis_transform


###################################
# Dataset
###################################


class BraTSDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
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
        self.train_transform = get_train_transform(spatial_size=spatial_size)
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
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_loader


# alias
BRATS = BraTSDataModule

###################################
# Inference
###################################


class OneHotDecoder(Transform):
    def __call__(self, mask):
        et = mask[:, 0, ...]
        tc = mask[:, 1, ...]
        wt = mask[:, 2, ...]
        out = torch.zeros_like(wt)
        out[et == 1] = 3
        out[torch.logical_and(tc == 1, et == 0)] = 2
        out[torch.logical_and(wt == 1, tc == 0)] = 1
        return out


class Inferer(object):
    def __init__(
        self,
        spatial_size=(128, 128, 128),
        post="logit-nested",
        write_dir=None,
        output_dtype=None,
        **kwargs,
    ) -> None:
        super().__init__()

        # inference method
        self.inferer = SlidingWindowInferer(roi_size=spatial_size, **kwargs)

        # postprocessing transforms
        if post == "logit-nested":
            self.post = Lambdad("input", lambda x: x)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "prob-nested":
            self.post = Lambdad("input", lambda x: x.sigmoid(dim=1))
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "one-hot-nested":
            self.post = Lambdad(
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

            self.post = Lambdad("input", func)
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

            self.post = Lambdad("input", func)
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        else:
            self.post = post

        # write to file
        self.output_dtype = output_dtype
        self.write_dir = write_dir

    def get_inferred(self, batch, model):
        batch["input"] = self.inferer(batch["input"], model)
        return batch

    def get_postprocessed(self, batch, model):
        batch = self.get_inferred(batch, model)
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
