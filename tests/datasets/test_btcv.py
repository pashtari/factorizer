# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from factorizer import datasets

# %% Sample
dm = datasets.BTCV(
    data_properties="/Users/pooya/Data/BTCV/RawData/dataset.json",
    spacing=(1.5, 1.5, 2.0),
    spatial_size=(96, 96, 96),
    num_splits=5,
    split=0,
    batch_size=1,
    num_workers=0,
    num_init_workers=0,
    num_replace_workers=0,
    cache_num=5,
    cache_rate=1.0,
    replace_rate=0.2,
    progress=True,
    copy_cache=True,
    seed=42,
)
dm.setup("fit")

train_ds = dm.train_set
val_ds = dm.val_set

train_dl = dm.train_dataloader()
val_dl = dm.val_dataloader()


# sample one image
sample = train_ds[0]
image = sample[0]["input"].detach().cpu().numpy()
mask = sample[0]["target"].detach().cpu().numpy()

print(f"image shape: {image.shape}")
print(f"mask shape: {mask.shape}")


# find a good slice for visualization
decoded_mask = mask.argmax(0)
s = decoded_mask.shape[-1]
num_uniques = []
for j in range(decoded_mask.shape[-1]):
    num_uniques.append(len(np.unique(decoded_mask[..., j].flatten())))

slc = (slice(None), slice(None), np.argmax(num_uniques))


# visualize image
fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(0, *slc)], cmap="gray", origin="lower")


# visualize mask
fig, ax = plt.subplots(dpi=200)
ax.imshow(image[(0, *slc)], "gray", origin="lower")
cmap = get_cmap("tab20")
for j in range(1, 14):
    masked = np.ma.masked_where(mask[(j, *slc)] == 0, mask[(j, *slc)])
    cmap_j = ListedColormap([cmap.colors[j - 1]])
    ax.imshow(masked, cmap_j, alpha=1, origin="lower")

