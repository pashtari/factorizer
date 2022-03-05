import sys

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from monai.data import CacheDataset
from monai.apps import CrossValidation
from monai.data import DataLoader

from .utils import wrap_class
from .dataset import wrap_dataset


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_properties,
        train_dataset_cls=CacheDataset,
        val_dataset_cls=CacheDataset,
        test_dataset_cls=None,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        vis_transform=None,
        num_splits=5,
        split=0,
        batch_size=2,
        num_workers=None,
        seed=42,
        **kwargs
    ):
        super().__init__()
        self.data_properties = data_properties
        self.train_dataset_cls = wrap_dataset(wrap_class(train_dataset_cls))
        self.val_dataset_cls = wrap_dataset(wrap_class(val_dataset_cls))
        test_dataset_cls = (
            val_dataset_cls if test_dataset_cls is None else test_dataset_cls
        )
        self.test_dataset_cls = wrap_dataset(wrap_class(test_dataset_cls))

        self.num_splits = num_splits
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shared_dataset_params = kwargs

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.vis_transform = vis_transform

        self.train_set = self.val_set = self.test_set = None

    def setup(self, stage):
        if stage in ("fit", "validate", None):
            # make training set
            train_cv = CrossValidation(
                self.train_dataset_cls,
                self.num_splits,
                self.seed,
                data_properties=self.data_properties,
                section="training",
                transform=self.train_transform,
                **self.shared_dataset_params
            )
            train_folds = [
                k for k in range(self.num_splits) if k != self.split
            ]
            self.train_set = train_cv.get_dataset(train_folds)

            # make validation set
            val_cv = CrossValidation(
                self.val_dataset_cls,
                self.num_splits,
                self.seed,
                data_properties=self.data_properties,
                section="validation",
                transform=self.val_transform,
                **self.shared_dataset_params
            )
            self.val_set = val_cv.get_dataset([self.split])

        if stage in ("test", "predict", None):
            # make test set
            self.test_set = self.val_dataset_cls(
                data_properties=self.data_properties,
                section="test",
                transform=self.test_transform,
                **self.shared_dataset_params
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_loader


class SmartDatasetCallback(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        trainer.datamodule.train_set.start()

    def on_train_epoch_end(self, trainer, pl_module, unused=None) -> None:
        trainer.datamodule.train_set.update_cache()

    def on_train_end(self, trainer, pl_module) -> None:
        trainer.datamodule.train_set.shutdown()
