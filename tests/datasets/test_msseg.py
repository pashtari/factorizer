import os


import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, ToTensord, Spacingd
from monai.utils.misc import first

from factorizer.data import msseg

# %%
test = [
    {
        "id": "013",
        "input": [
            "/Users/pooya/Data/MSSEG-2/training/013/flair_time01_on_middle_space.nii.gz",
            "/Users/pooya/Data/MSSEG-2/training/013/flair_time02_on_middle_space.nii.gz",
        ],
    }
]
dm = msseg.MSSEG(
    data=None,
    test=test,
    spacing=(0.6, 0.6, 0.6),
    spatial_size=(96, 96, 96),
    num_workers=0,
    num_splits=5,
    split=0,
    batch_size=1,
    seed=42,
)
dm.setup("test")
batch = first(dm.test_dataloader())

# %%
dm = msseg.MSSEG(
    data="/Users/pooya/Data/MSSEG-2/training",
    test=None,
    spacing=(0.6, 0.6, 0.6),
    spatial_size=(96, 96, 96),
    num_workers=0,
    num_splits=5,
    split=0,
    batch_size=1,
    seed=42,
)
dm.setup("validate")
batch = first(dm.val_dataloader())

# %% test interpolate
interp = msseg.Interpolate("input", spacing=(0.6, 0.6, 0.6), mode="trilinear")
batch1 = interp.transform(batch)
batch2 = interp.inverse_transform(batch1)


# %% visualize before and after interpolation
fig, ax = plt.subplots(dpi=200)
ax.imshow(batch["input"][0, 0, :, :, 127])
plt.show()

fig, ax = plt.subplots(dpi=200)
ax.imshow(batch1["input"][0, 0, :, :, 208])
plt.show()

fig, ax = plt.subplots(dpi=200)
ax.imshow(batch2["input"][0, 0, :, :, 127])
plt.show()


fig, ax = plt.subplots(dpi=200)
ax.imshow((batch2["input"] - batch["input"])[0, 0, :, :, 127])
plt.show()
