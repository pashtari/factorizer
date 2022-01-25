import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from factorizer import datasets

# %% Sample
dm = datasets.BRATS(
    data_properties="/Users/pooya/Data/Decathlon/Task01_BrainTumour/dataset.json",
    spacing=(1, 1, 1),
    spatial_size=(128, 128, 128),
    num_splits=5,
    split=0,
    batch_size=1,
    num_workers=0,
    cache_num=0,
    cache_rate=1,
    progress=True,
    copy_cache=True,
    seed=42,
)
dm.setup("fit")

train_ds = dm.train_set
val_ds = dm.val_set

train_dl = dm.train_dataloader()
val_dl = dm.val_dataloader()

# pick one image from DecathlonDataset to visualize and check the 4 channels
sample = val_ds[1]
image = sample["input"].detach().cpu()
mask = sample["target"].detach().cpu()

# find a good slice for visualization
slc = (slice(None), slice(None), np.argmax(mask[0].sum((0, 1))))


# visualize image
fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(0, *slc)], cmap="gray", origin="lower")


# visualize mask
fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(0, *slc)], "gray", origin="lower")
cmap = get_cmap("tab10")
for j in range(2, -1, -1):
    masked = np.ma.masked_where(mask[(j, *slc)] == 0, mask[(j, *slc)])
    cmap_j = ListedColormap([cmap.colors[j]])
    ax.imshow(masked, cmap_j, alpha=1, origin="lower")

