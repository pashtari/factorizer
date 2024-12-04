from collections.abc import Hashable, Mapping

import numpy as np
import torch
from monai.data import load_decathlon_datalist
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.enums import TransformBackends


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


class BraTSOneHotEncoder(Transform):
    """
    Convert labels to multi channels based on BraTS (2023-2024) classes:

    class 0: background (excluded),
    class 1: necrotic and non-enhancing tumor (NCR/NET)
    class 2: edema (ED)
    class 3: enhancing tumor (ET)

    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        ed, ncr, et = 2, 1, 3
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [
            (img == et),
            (img == et) | (img == ncr),
            (img == et) | (img == ncr) | (img == ed),
        ]
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )


class BraTSOneHotEncoderd(MapTransform):
    """Dictionary-based wrapper of `BraTSOneHotEncoder`."""

    backend = BraTSOneHotEncoder.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = BraTSOneHotEncoder()

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
