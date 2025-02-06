from monai.data import load_decathlon_datalist


def load_datalist(datalist_path, data_dir=None, key="training", fold=None, section=None):
    """
    Load a datalist for the FIVES dataset, with optional support for cross-validation folds.

    Args:
        datalist_path (str): Path to the JSON file containing the dataset metadata.
        data_dir (str): Base directory for the dataset (optional).
        key (str): Key in the JSON file to access data (default: "training").
        fold (int): Fold number to use for cross-validation (optional).
        section (str): One of ["training", "validation"] to specify which part of the data to load.

    Returns:
        list: A list of data entries for the specified section.
    """
    data_list = load_decathlon_datalist(
        datalist_path, data_list_key=key, base_dir=data_dir
    )
    if fold is None:
        return data_list
    elif section in ("training", "train"):
        return [x for x in data_list if x["fold"] != fold]
    elif section in ("validation", "val"):
        return [x for x in data_list if x["fold"] == fold]
    else:
        raise ValueError(
            "When `fold` is provided, `section` must be one of ['training', 'validation']."
        )
