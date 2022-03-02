import yaml

from torch import nn, optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from monai.losses import DiceCELoss

import factorizer as ft
from factorizer import datasets
from factorizer.utils.lightning import SemanticSegmentation
from factorizer.utils.losses import DeepSuprLoss
from factorizer.utils.metrics import DiceMetric, HausdorffDistanceMetric
from factorizer.utils.schedulers import WarmupCosineSchedule
from factorizer.utils.helpers import SaveValResults


def lambda_constructor(loader, node):
    lambda_expr = "lambda " + loader.construct_scalar(node)
    return eval(lambda_expr)


def get_constructor(obj):
    """Get constructor for an object."""

    def constructor(loader, node):
        if isinstance(node, yaml.nodes.ScalarNode):
            if node.value:
                out = obj(loader.construct_scalar(node))
            else:
                out = obj
        elif isinstance(node, yaml.nodes.SequenceNode):
            out = obj(*loader.construct_sequence(node, deep=True))
        elif isinstance(node, yaml.nodes.MappingNode):
            out = obj(**loader.construct_mapping(node, deep=True))

        return out

    return constructor


Loader = yaml.SafeLoader


# general
Loader.add_constructor("!eval", get_constructor(eval))
Loader.add_constructor("!lambda", lambda_constructor)

# data modules
Loader.add_constructor(
    "!BraTSDataModule", get_constructor(datasets.BraTSDataModule)
)
Loader.add_constructor("!BraTSInferer", get_constructor(datasets.BraTSInferer))


# tasks
Loader.add_constructor(
    "!SemanticSegmentation", get_constructor(SemanticSegmentation)
)


# layers and blocks
Loader.add_constructor("!Conv3d", get_constructor(nn.Conv3d))
Loader.add_constructor("!ConvTranspose3d", get_constructor(nn.ConvTranspose3d))
Loader.add_constructor("!GroupNorm", get_constructor(nn.GroupNorm))
Loader.add_constructor("!LeakyReLU", get_constructor(nn.LeakyReLU))
Loader.add_constructor("!ReLU", get_constructor(nn.ReLU))
Loader.add_constructor("!GELU", get_constructor(nn.GELU))
Loader.add_constructor("!Same", get_constructor(ft.Same))
Loader.add_constructor("!DoubleConv", get_constructor(ft.DoubleConv))
Loader.add_constructor("!BasicBlock", get_constructor(ft.BasicBlock))
Loader.add_constructor(
    "!PreActivationBlock", get_constructor(ft.PreActivationBlock)
)
Loader.add_constructor(
    "!FactorizerSubblock", get_constructor(ft.FactorizerSubblock)
)
Loader.add_constructor("!Matricize", get_constructor(ft.Matricize))
Loader.add_constructor("!SWMatricize", get_constructor(ft.SWMatricize))
Loader.add_constructor("!NMF", get_constructor(ft.NMF))
Loader.add_constructor("!MLP", get_constructor(ft.MLP))
Loader.add_constructor(
    "!FastSelfAttention", get_constructor(ft.FastSelfAttention)
)
Loader.add_constructor("!ablate", get_constructor(ft.ablate))


# models
Loader.add_constructor("!UNet", get_constructor(ft.UNet))
Loader.add_constructor(
    "!SegmentationFactorizer", get_constructor(ft.SegmentationFactorizer)
)


# losses
Loader.add_constructor("!DiceCELoss", get_constructor(DiceCELoss))
Loader.add_constructor("!DeepSuprLoss", get_constructor(DeepSuprLoss))


# optimizers and scheduler
Loader.add_constructor("!AdamW", get_constructor(optim.AdamW))
Loader.add_constructor(
    "!WarmupCosineSchedule", get_constructor(WarmupCosineSchedule)
)


# metrics
Loader.add_constructor("!DiceMetric", get_constructor(DiceMetric))
Loader.add_constructor(
    "!HausdorffDistanceMetric", get_constructor(HausdorffDistanceMetric)
)


# callbacks
Loader.add_constructor(
    "!LearningRateMonitor", get_constructor(LearningRateMonitor)
)
Loader.add_constructor("!ModelCheckpoint", get_constructor(ModelCheckpoint))
Loader.add_constructor(
    "!TensorBoardLogger", get_constructor(TensorBoardLogger)
)
Loader.add_constructor("!SaveValResults", get_constructor(SaveValResults))


def read_config(path):
    with open(path, "rb") as file:
        config = yaml.load(file, Loader)

    return config

