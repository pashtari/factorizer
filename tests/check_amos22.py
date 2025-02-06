import os
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    CropForegroundd,
    Orientationd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    Spacingd,
    EnsureTyped,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandFlipd,
)
from monai.data import DataLoader, Dataset
from monai.utils import first

# Define deterministic transformations
deterministic_transforms = [
    LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    Spacingd(
        keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")
    ),
    EnsureTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
]

# Define random transformations
random_transforms = [
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(128, 128, 128),
        pos=1,
        neg=1,
        num_samples=1,
    ),
    RandAffined(
        keys=["image", "label"],
        prob=0.2,
        spatial_size=(128, 128, 128),
        rotate_range=(0.26, 0.26, 0.26),
        scale_range=(0.2, 0.2, 0.2),
        mode=("bilinear", "nearest"),
        cache_grid=True,
        padding_mode="border",
    ),
    RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
    RandGaussianSmoothd(
        keys="image", prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)
    ),
    RandScaleIntensityd(keys="image", prob=0.2, factors=0.3),
    RandShiftIntensityd(keys="image", prob=0.2, offsets=0.1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
]

# Combine deterministic and random transforms
train_transforms = Compose(deterministic_transforms + random_transforms)

# Load and transform the data
data_dicts = [
    {
        "image": "/Volumes/T7/data/AMOS22/imagesTr/amos_0011.nii.gz",
        "label": "/Volumes/T7/data/AMOS22/labelsTr/amos_0011.nii.gz",
    }
]
dataset = Dataset(data=data_dicts, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=1)

# Get a sample
sample = first(dataloader)

image = sample["image"][0, 0]  # Get the first sample's image
label = sample["label"][0, 0]  # Get the first sample's label

# Visualize the middle slice of the 3D image
slice_idx = image.shape[-1] // 2

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, slice_idx], cmap="gray")
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(label[:, :, slice_idx], cmap="jet", alpha=0.5)
plt.title("Label Overlay")
plt.axis("off")

plt.show()
