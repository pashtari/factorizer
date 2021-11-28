import matplotlib.pyplot as plt

from factorizer import datasets

# %% Sample
dm = datasets.BRATS(
    root_dir="/Users/pooya/Data/Decathlon/Task01_BrainTumour",
    num_splits=5,
    split=0,
    num_workers=0,
    batch_size=1,
    cache_num=0,
    seed=42,
)
dm.setup("fit")

train_ds = dm.train_set
val_ds = dm.val_set

train_dl = dm.train_dataloader()
val_dl = dm.val_dataloader()

# pick one image from DecathlonDataset to visualize and check the 4 channels
sample = train_ds[0]
image = sample["input"].detach().cpu()
mask = sample["target"].detach().cpu()

print(f"image shape: {image.shape}")
plt.figure("input", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(image[i, :, :, 70], cmap="gray")
plt.show()
# also visualize the 3 channels label corresponding to this image
print(f"mask shape: {mask.shape}")
plt.figure("mask", (18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"mask channel {i}")
    plt.imshow(mask[i, :, :, 70])
plt.show()

# %%
