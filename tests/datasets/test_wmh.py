import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from factorizer import datasets

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# %% Sample
dm = datasets.WMH(
    data_properties="/Users/pooya/Data/WMH/dataset.json",
    spacing=(1.0, 1.0, 1.0),
    spatial_size=(128, 128, 128),
    num_splits=5,
    split=0,
    batch_size=1,
    num_workers=0,
    cache_num=0,
    cache_rate=0,
    progress=True,
    copy_cache=True,
    seed=42,
)
dm.setup("fit")

train_ds = dm.train_set
val_ds = dm.val_set

train_dl = dm.train_dataloader()
val_dl = dm.val_dataloader()

# pick one image to visualize and check the 2 channels
sample = train_ds[1]
image = sample[0]["input"].detach().cpu()
mask = sample[0]["target"].detach().cpu()

# find a good slice for visualization
slc = (slice(None), slice(None), np.argmax(mask[0].sum((0, 1))))


# visualize image
fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(0, *slc)], cmap="gray", origin="lower")

fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(1, *slc)], cmap="gray", origin="lower")


# visualize mask
fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(1, *slc)], "gray", origin="lower")
cmap = get_cmap("tab10")
masked = np.ma.masked_where(mask[(0, *slc)] == 0, mask[(0, *slc)])
cmap_j = ListedColormap([cmap.colors[0]])
ax.imshow(masked, cmap_j, alpha=1, origin="lower")

