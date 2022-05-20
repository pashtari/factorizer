import yaml

from torch import nn, optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from monai.losses import DiceCELoss

from .factorizer import SegmentationFactorizer, FactorizerSubblock, ablate
from .unet import UNet, Same, DoubleConv, BasicBlock, PreActivationBlock
from .layers import MLP, FastSelfAttention
from .factorization import NMF, Matricize, SWMatricize
from .datasets import BraTSDataModule, BraTSInferer
from .utils.lightning import SemanticSegmentation
from .utils.losses import DeepSuprLoss
from .utils.metrics import DiceMetric, HausdorffDistanceMetric
from .utils.schedulers import WarmupCosineSchedule
from .utils.helpers import SaveValResults


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
Loader.add_constructor("!BraTSDataModule", get_constructor(BraTSDataModule))
Loader.add_constructor("!BraTSInferer", get_constructor(BraTSInferer))


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
Loader.add_constructor("!Same", get_constructor(Same))
Loader.add_constructor("!DoubleConv", get_constructor(DoubleConv))
Loader.add_constructor("!BasicBlock", get_constructor(BasicBlock))
Loader.add_constructor(
    "!PreActivationBlock", get_constructor(PreActivationBlock)
)
Loader.add_constructor(
    "!FactorizerSubblock", get_constructor(FactorizerSubblock)
)
Loader.add_constructor("!Matricize", get_constructor(Matricize))
Loader.add_constructor("!SWMatricize", get_constructor(SWMatricize))
Loader.add_constructor("!NMF", get_constructor(NMF))
Loader.add_constructor("!MLP", get_constructor(MLP))
Loader.add_constructor(
    "!FastSelfAttention", get_constructor(FastSelfAttention)
)
Loader.add_constructor("!ablate", get_constructor(ablate))


# models
Loader.add_constructor("!UNet", get_constructor(UNet))
Loader.add_constructor(
    "!SegmentationFactorizer", get_constructor(SegmentationFactorizer)
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


def read_config(path, loader=Loader):
    with open(path, "rb") as file:
        config = yaml.load(file, loader)

    return config

