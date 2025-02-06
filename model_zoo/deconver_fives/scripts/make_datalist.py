"""
This script processes the FIVES dataset to generate a datalist JSON file in a Decathlon-style format.

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
    image_list = sorted(glob.glob("**/Original/*.png", recursive=True))

    # Make training set
    datalist = []
    for img in image_list:
        id_ = os.path.dirname(img) + "_" + os.path.basename(img).replace(".png", "")
        mask = img.replace("/Original", "/Ground truth")
        sample = {"id": id_, "image": img, "label": mask}
        datalist.append(sample)

    return datalist


def roi_area(mask):
    """Computes ROI area from a segmentation mask."""
    roi_num_pixels = torch.sum(mask[..., -1] == 255)
    return roi_num_pixels.item()


def add_area(datalist):
    """Adds roi areas to the given datalist."""
    transform = transforms.Compose(
        [
            transforms.CopyItemsd(keys="label", names="area"),
            transforms.LoadImaged(keys="area"),
            transforms.Lambdad(keys="area", func=roi_area),
        ]
    )

    dataset = Dataset(datalist, transform)
    updated_datalist = list(dataset)

    return updated_datalist


def stratified_kfold(datalist, num_bins=5, num_folds=5):
    """Makes a stratified k-fold cross-validation based on roi area."""

    # Quantize roi areas
    roi_areas = [x["area"] for x in datalist]
    _, bins, _ = plt.hist(roi_areas, bins=num_bins)
    y = np.digitize(roi_areas, bins[:-1])

    # Stratified K-fold based on quantized roi areas
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    skf.get_n_splits(None, y)

    updated_datalist = []
    for fold, (_, val_idx) in enumerate(skf.split(datalist, y)):
        for j in val_idx:
            sample = datalist[j]
            del sample["area"]
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

    # Create datalists for FIVES dataset
    train_list = make_datalist()

    # Add roi areas to training list
    train_list = add_area(train_list)

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
    export_datalist(train_list, [], file_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare datalist for FIVES dataset.")
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
