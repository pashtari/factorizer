from monai.data import load_decathlon_datalist


def load_datalist(datalist_path, data_dir=None, key="training", fold=None, section=None):
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
