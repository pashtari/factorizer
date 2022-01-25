#%%
import os
import pickle


import torch
from pytorch_lightning import Trainer

import factorizer as ft
from factorizer.utils.helpers import read_config
from factorizer.utils.lightning import Model
from pytorch_lightning import Trainer

# ft.factorizer.MLP = ft.MLP
# ft.factorization.operations.SwinMetricize = ft.SWMatricize
# ft.factorizer.NMF = ft.NMF

method = "nmf"
config = f"/Users/pooya/OneDrive - KU Leuven/PhD_thesis/research/factorizer/bin/configs/ablations/config_brats_fold0_{method}.py"
config = read_config(config)
model = Model.load_from_checkpoint(
    **config["test"]["checkpoint"],
    # **config["model"],
    # **config["optimization"],
)

# matricize = model.net.encoder.blocks[0].blocks[0].blocks["nmf"].fn[1].tensorize

del (
    config["training"]["gpus"],
    config["training"]["num_nodes"],
    config["training"]["accelerator"],
)
trainer = Trainer(**config["training"])
trainer.model = model
# ckpt_name = os.path.basename(
#     config["test"]["checkpoint"]["checkpoint_path"]
# ).split(".")[0]
# ckpt_dir = os.path.dirname(config["test"]["checkpoint"]["checkpoint_path"])
# ckpt_path = f"{ckpt_dir}/{ckpt_name}_v2.ckpt"
ckpt_path = config["test"]["checkpoint"]["checkpoint_path"]
trainer.save_checkpoint(ckpt_path)

#%%
from factorizer.utils.helpers import read_config
from factorizer.utils.lightning import Model

method = "mlp"
config = f"/Users/pooya/OneDrive - KU Leuven/PhD_thesis/research/factorizer/bin/configs/ablations/config_brats_fold0_{method}.py"
config = read_config(config)

# ckpt_name = os.path.basename(
#     config["test"]["checkpoint"]["checkpoint_path"]
# ).split(".")[0]
# ckpt_dir = os.path.dirname(config["test"]["checkpoint"]["checkpoint_path"])
# ckpt_path = f"{ckpt_dir}/{ckpt_name}_v2.ckpt"
ckpt_path = config["test"]["checkpoint"]["checkpoint_path"]
model = Model.load_from_checkpoint(
    ckpt_path, **config["model"], **config["optimization"],
)

# %%
