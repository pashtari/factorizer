"""
This script processes the BraTS23-GLI dataset to generate a datalist JSON file in a Decathlon-style format.

Usage:
    $ python make_datalist.py --data_dir /path/to/data --save_path /path/to/datalist --num_bins <num_bins> --num_folds <num_folds>

Arguments:
    - data_dir: Path to the dataset directory (required).
    - save_path: Path to save the datalist JSON file (default: '../configs/datalist.json').
    - num_bins: Number of bins for stratified K-fold (default: 5).
    - num_folds: Number of folds for stratified K-fold (default: 5).

"""

import argparse
import glob
import os
import json

import numpy as np
import torch
from monai.data import Dataset
from monai import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def make_datalist():
    """Makes the train and test datalist."""
    train_folders = sorted(glob.glob("*Training*/*"))
    test_folders = sorted(glob.glob("*Validation*/*"))

    # Make training set
    train_datalist = []
    for folder in train_folders:
        id_ = os.path.basename(folder)
        t1n = glob.glob(os.path.join(folder, "*t1n.nii.gz"))[0]
        t1c = glob.glob(os.path.join(folder, "*t1c.nii.gz"))[0]
        t2w = glob.glob(os.path.join(folder, "*t2w.nii.gz"))[0]
        t2f = glob.glob(os.path.join(folder, "*t2f.nii.gz"))[0]
        seg = glob.glob(os.path.join(folder, "*seg.nii.gz"))[0]
        sample = {"id": id_, "image": [t1n, t1c, t2w, t2f], "label": seg}
        train_datalist.append(sample)

    # Make test set
    test_list = []
    for folder in test_folders:
        id_ = os.path.basename(folder)
        t1n = glob.glob(os.path.join(folder, "*t1n.nii.gz"))[0]
        t1c = glob.glob(os.path.join(folder, "*t1c.nii.gz"))[0]
        t2w = glob.glob(os.path.join(folder, "*t2w.nii.gz"))[0]
        t2f = glob.glob(os.path.join(folder, "*t2f.nii.gz"))[0]
        sample = {"id": id_, "image": [t1n, t1c, t2w, t2f]}
        test_list.append(sample)

    test_list = sorted(test_list, key=lambda x: x["id"])

    return train_datalist, test_list


def lesion_volume(mask):
    """Computes lesion volume from a segmentation mask."""
    lesion_num_voxels = torch.sum(
        mask == 3
    )  # Assuming enhancing tumor (class 3) is more important
    voxel_size = np.prod(mask.meta["pixdim"][1:4])
    lesion_vol = lesion_num_voxels * voxel_size
    return lesion_vol.item()


def add_volume(datalist):
    """Adds lesion volumes to the given datalist."""
    transform = transforms.Compose(
        [
            transforms.CopyItemsd(keys="label", names="volume"),
            transforms.LoadImaged(keys="volume"),
            transforms.Lambdad(keys="volume", func=lesion_volume),
        ]
    )

    dataset = Dataset(datalist, transform)
    updated_datalist = list(dataset)

    return updated_datalist


def stratified_kfold(datalist, num_bins=5, num_folds=5):
    """Makes a stratified k-fold cross-validation based on lesion volume."""

    # Quantize lesion volumes
    lesion_volumes = [x["volume"] for x in datalist]
    _, bins, _ = plt.hist(lesion_volumes, bins=num_bins)
    y = np.digitize(lesion_volumes, bins[:-1])

    # Stratified K-fold based on quantized lesion volumes
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    skf.get_n_splits(None, y)

    updated_datalist = []
    for fold, (_, val_idx) in enumerate(skf.split(datalist, y)):
        for j in val_idx:
            sample = datalist[j]
            del sample["volume"]
            updated_datalist.extend([{**sample, "fold": fold}])

    updated_datalist = sorted(updated_datalist, key=lambda x: x["id"])

    return updated_datalist


def export_datalist(train_list, test_list, file_path):
    """Export the dataset splits to a JSON file."""
    data_list = {"training": train_list, "test": test_list}
    with open(file_path, "w") as f:
        json.dump(data_list, f)
    print(f"Datalist saved to {file_path}.")


def main(args):
    # Set working directory to data directory
    os.chdir(args.data_dir)

    # Create datalists for BraTS'23-GLI dataset
    train_list, test_list = make_datalist()

    # Add lesion volumes to training list
    train_list = add_volume(train_list)

    # Apply Stratified K-Fold to training list
    train_list = stratified_kfold(
        train_list, num_bins=args.num_bins, num_folds=args.num_folds
    )

    # Define path to save datalist
    if args.save_path is None:
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "configs", "datalist.json"
        )
    else:
        file_path = args.save_path
    file_path = os.path.abspath(file_path)

    # Export datalist to JSON
    export_datalist(train_list, test_list, file_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare datalist for BraTS'23 dataset.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the datalist JSON file"
    )
    parser.add_argument(
        "--num_bins", type=int, default=5, help="Number of bins for stratified K-fold"
    )
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of folds for stratified K-fold"
    )
    args = parser.parse_args()
    main(args)
