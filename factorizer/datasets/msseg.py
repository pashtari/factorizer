import os
from typing import Sequence
import copy

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Orientationd,
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd,
    Lambdad,
    ToTensord,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandBiasFieldd,
    Activationsd,
    AsDiscreted,
)
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch
from monai.utils.misc import first

from .utils import SaveImaged


# %% Dataset
def get_data(root_dir):
    cases = []
    for f in os.scandir(root_dir):
        if f.is_dir() and not f.name.startswith("."):
            case_dict = {"id": f.name, "input": {}, "target": None}
            for i in os.scandir(f.path):
                if "flair_time01_on_middle_space.nii" in i.name:
                    case_dict["input"]["flair01"] = i.path
                elif "flair_time02_on_middle_space.nii" in i.name:
                    case_dict["input"]["flair02"] = i.path
                elif "ground_truth.nii" in i.name:
                    case_dict["target"] = i.path

            case_dict["input"] = [
                case_dict["input"]["flair01"],
                case_dict["input"]["flair02"],
            ]
            cases.append(case_dict)

    cases = sorted(cases, key=lambda x: x["id"])
    return cases


# %% Transform
def get_train_transform(spacing=(1.0, 1.0, 1.0), spatial_size=(128, 128, 128)):
    transforms = [
        LoadImaged(["input", "target"]),
        AddChanneld("target"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(["input", "target"], source_key="input"),
        NormalizeIntensityd("input", nonzero=True, channel_wise=True),
        Spacingd(
            keys=["input", "target"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        RandCropByPosNegLabeld(
            keys=["input", "target"],
            label_key="target",
            spatial_size=spatial_size,
            pos=2,
            neg=1,
            num_samples=1,
            image_key="input",
            image_threshold=0,
        ),
        RandAffined(
            ["input", "target"],
            prob=0.15,
            spatial_size=spatial_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "bilinear"),
            as_tensor_output=False,
        ),
        RandFlipd(["input", "target"], prob=0.5, spatial_axis=0),
        RandFlipd(["input", "target"], prob=0.5, spatial_axis=1),
        RandFlipd(["input", "target"], prob=0.5, spatial_axis=2),
        RandGaussianNoised("input", prob=0.15, std=0.1),
        RandGaussianSmoothd(
            "input",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        RandScaleIntensityd("input", prob=0.15, factors=0.3),
        RandShiftIntensityd("input", prob=0.15, offsets=0.1),
        RandBiasFieldd("input", prob=0.15, degree=3, coeff_range=(0.0, 0.1)),
        RandAdjustContrastd("input", prob=0.15, gamma=(0.7, 1.5)),
        Lambdad("target", func=lambda x: np.where(x >= 0.5, 1.0, 0.0)),
        ToTensord(["input", "target"]),
    ]
    train_transform = Compose(transforms)
    return train_transform


def get_val_transform():
    transforms = [
        LoadImaged(["input", "target"], allow_missing_keys=True),
        AddChanneld("target", allow_missing_keys=True),
        NormalizeIntensityd("input", nonzero=True, channel_wise=True),
        ToTensord(["input", "target"], allow_missing_keys=True),
    ]
    val_transform = Compose(transforms)
    return val_transform


def get_test_transform():
    return get_val_transform()


def get_vis_transform(spacing=(1.0, 1.0, 1.0)):
    transforms = [
        LoadImaged(["input", "target"]),
        AddChanneld("target"),
        Spacingd(
            keys=["input", "target"],
            pixdim=spacing,
            mode=("bilinear", "bilinear"),
        ),
        Lambdad("target", func=lambda x: np.where(x >= 0.5, 1.0, 0.0)),
        ToTensord(["input", "target"]),
    ]
    vis_transform = Compose(transforms)
    return vis_transform


class MSSEGDataModule(LightningDataModule):
    def __init__(
        self,
        data=None,
        test=None,
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
        num_workers=2,
        batch_size=2,
        num_splits=5,
        split=0,
        seed=42,
    ):
        super().__init__()
        self.data = data
        self.test = test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        self.kfold = KFold(num_splits, random_state=seed, shuffle=True)

        # get training transform
        self.train_transform = get_train_transform(spacing, spatial_size)
        # get validation transform
        self.val_transform = get_val_transform()
        # get test transform
        self.test_transform = get_test_transform()
        # get visualization transform
        self.vis_transform = get_vis_transform(spacing)

        # get train/validation subjects
        if isinstance(data, str):
            self.data_cases = get_data(data)
        elif isinstance(data, Sequence):
            self.data_cases = data

        # get test subjects
        if isinstance(test, str):
            self.test_cases = get_data(test)
        elif isinstance(test, Sequence):
            self.test_cases = test

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage in ("fit", "validate", None):
            self.data_cases = np.array(self.data_cases)
            # split subjects into training and validation
            train_index, val_index = list(self.kfold.split(self.data_cases))[
                self.split
            ]
            # make training set
            self.train_cases = list(self.data_cases[train_index])
            self.train_set = Dataset(
                self.train_cases, transform=self.train_transform
            )
            # make validation set
            self.val_cases = list(self.data_cases[val_index])
            self.val_set = Dataset(
                self.val_cases, transform=self.val_transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage in ("test", "predict", None):
            # make test set
            self.test_set = Dataset(
                self.test_cases, transform=self.test_transform
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
MSSEG = MSSEGDataModule


# # %% Inference
# def move_to(d, *args, **kwargs):
#     if isinstance(d, (list, tuple, set, frozenset)):
#         d = [move_to(v, *args, **kwargs) for v in d]
#     elif isinstance(d, dict):
#         d = {k: move_to(v, *args, **kwargs) for k, v in d.items()}
#     elif isinstance(d, torch.Tensor):
#         d = d.to(*args, **kwargs)

#     return d


# def transform_batch(batch, transform, **kwargs):
#     batch = Dataset(batch, transform)
#     batch = DataLoader(batch, **kwargs)
#     return batch


# class Inferer(object):
#     def __init__(
#         self,
#         spacing=(1.0, 1.0, 1.0),
#         spatial_size=(128, 128, 128),
#         write_dir=None,
#         num_workers=0,
#         output_type="one-hot",
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.num_workers = num_workers
#         assert output_type in {
#             "logit",
#             "prob",
#             "one-hot",
#             "class",
#         }, f'output_type must be "logit", "prob", "one-hot", or "class"; got {output_type}'
#         self.output_type = output_type
#         self.device = None

#         # preprocessing transforms
#         self.spacing = Spacingd(keys="input", pixdim=spacing, mode="bilinear")

#         # inference method
#         self.inferer = SlidingWindowInferer(roi_size=spatial_size, **kwargs)

#         # postprocessing transforms
#         self.sigmoid = Activationsd("input", sigmoid=True)
#         self.as_discrete = AsDiscreted("input", threshold_values=True)

#         # write to file
#         self.write_dir = write_dir

#     def get_preprocessed(self, batch, model):
#         self.device = batch["input"].device
#         batch = move_to(batch, device="cpu")
#         batch = decollate_batch(batch)
#         batch = transform_batch(
#             batch,
#             self.spacing,
#             num_workers=self.num_workers,
#             batch_size=len(batch),
#         )
#         batch = first(batch)
#         return batch

#     def get_inferred(self, batch, model):
#         batch = self.get_preprocessed(batch, model)
#         batch = move_to(batch, device=self.device)
#         batch["input"] = self.inferer(batch["input"], model)
#         return batch

#     def get_postprocessed(self, batch, model):
#         batch = self.get_inferred(batch, model)
#         batch = move_to(batch, device="cpu")
#         batch = decollate_batch(batch)
#         for j, sample in enumerate(batch):
#             trans = sample["input_transforms"]
#             trans = [t for t in trans if t["class"] == "Spacingd"]
#             batch[j]["input_transforms"] = trans[-1:]

#         batch = transform_batch(
#             batch,
#             self.spacing.inverse,
#             num_workers=self.num_workers,
#             batch_size=len(batch),
#         )
#         batch = first(batch)
#         batch = move_to(batch, device=self.device)

#         if self.output_type == "prob":
#             batch = self.sigmoid(batch)

#         elif self.output_type in ("one-hot", "class"):
#             batch = self.sigmoid(batch)
#             batch = self.as_discrete(batch)

#         return batch

#     def write(self, batch, write_dir):
#         if write_dir is not None:
#             if self.output_type in ("logit", "prob"):
#                 output_dtype = np.float32
#             else:
#                 output_dtype = np.uint8

#             batch["input_meta_dict"]["filename_or_obj"] = batch["id"]
#             self.save = SaveImaged(
#                 "input",
#                 output_dir=write_dir,
#                 output_dtype=output_dtype,
#                 output_postfix="",
#                 save_batch=True,
#                 print_log=True,
#             )
#             self.save(batch)

#     def __call__(self, batch, model):
#         batch = self.get_postprocessed(batch, model)
#         self.write(batch, self.write_dir)
#         return batch["input"]


# %% Inference
class Interpolate(object):
    def __init__(self, key, spacing, **kwargs):
        super().__init__()
        self.key = key
        self.meta_key = f"{key}_meta_dict"
        self.spacing = torch.tensor(spacing)
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
        write_dir=None,
        output_type="one-hot",
        **kwargs,
    ):
        super().__init__()

        assert output_type in {
            "logit",
            "prob",
            "one-hot",
            "class",
        }, f'output_type must be "logit", "prob", "one-hot", or "class"; got {output_type}'
        self.output_type = output_type
        self.device = None

        # downsampling
        self.interp = Interpolate("input", spacing=spacing, mode="trilinear")

        # inference method
        self.inferer = SlidingWindowInferer(roi_size=spatial_size, **kwargs)

        # postprocessing transforms
        self.sigmoid = Activationsd("input", sigmoid=True)
        self.as_discrete = AsDiscreted("input", threshold_values=True)

        # write to file
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

        if self.output_type == "prob":
            batch = self.sigmoid(batch)

        elif self.output_type in ("one-hot", "class"):
            batch = self.sigmoid(batch)
            batch = self.as_discrete(batch)

        return batch

    def write(self, batch, write_dir):
        if write_dir is not None:
            if self.output_type in ("logit", "prob"):
                output_dtype = np.float32
            else:
                output_dtype = np.uint8

            batch["input_meta_dict"]["filename_or_obj"] = batch["id"]
            self.save = SaveImaged(
                "input",
                output_dir=write_dir,
                output_dtype=output_dtype,
                output_postfix="",
                save_batch=True,
                print_log=True,
            )
            self.save(batch)

    def __call__(self, batch, model):
        batch = self.get_postprocessed(batch, model)
        self.write(batch, self.write_dir)
        return batch["input"]
