import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    SaveImaged,
)
from monai.data import CacheDataset

from factorizer import load_properties


# %% Sample
def lits_transform():
    transforms = [
        LoadImaged(["image", "label"]),
        AddChanneld(["image", "label"]),
        CropForegroundd(["image", "label"], source_key="image"),
        SaveImaged(
            keys="image",
            output_dir="./fgcropped",
            output_postfix="",
            output_ext=".nii",
            separate_folder=False,
            output_dtype=np.int16,
        ),
        SaveImaged(
            keys="label",
            output_dir="./fgcropped",
            output_postfix="",
            output_ext=".nii",
            separate_folder=False,
            output_dtype=np.uint8,
        ),
    ]
    train_transform = Compose(transforms,)
    return train_transform


data = load_properties("/Users/pooya/Data/LiTS2017/dataset.json")
ds = CacheDataset(
    data["training"],
    transform=lits_transform(),
    num_workers=0,
    cache_num=0,
    cache_rate=1.0,
    progress=True,
    copy_cache=True,
)

[ds[n] for n in range(len(ds))]
# # sample one image
# sample = ds[0]
# image = sample[0]["input"].detach().cpu().numpy()
# mask = sample[0]["target"].detach().cpu().numpy()

# print(f"image shape: {image.shape}")
# print(f"mask shape: {mask.shape}")


# # find a good slice for visualization
# slc = (slice(None), slice(None), np.argmax(mask[2].sum((0, 1))))


# # visualize image
# fig, ax = plt.subplots(dpi=200)
# ax.imshow(image[(0, *slc)], cmap="gray", origin="lower")


# # visualize mask
# fig, ax = plt.subplots(dpi=200)
# ax.imshow(image[(0, *slc)], "gray", origin="lower")
# cmap = get_cmap("tab10")
# for j in range(1, 3):
#     masked = np.ma.masked_where(mask[(j, *slc)] == 0, mask[(j, *slc)])
#     cmap_j = ListedColormap([cmap.colors[j - 1]])
#     ax.imshow(masked, cmap_j, alpha=1, origin="lower")


# %%
