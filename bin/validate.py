import os
from argparse import ArgumentParser, Namespace

import torch
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything

from factorizer.utils.helpers import wrap_class, read_config, move_to
from factorizer.utils.lightning import Model


seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
print("cuda" if torch.cuda.is_available() else "cpu")

# os.chdir(os.path.dirname(__file__))


def main(args):
    # get config
    config = read_config(args.config)

    # load model
    if (
        "checkpoint" in config["test"]
        and "checkpoint_path" in config["test"]["checkpoint"]
    ):
        model = Model.load_from_checkpoint(
            **config["test"]["checkpoint"],
            strict=False,
            **config["model"],
            **config["optimization"]
        )
    else:
        model = Model(**config["model"], **config["optimization"])

    # load data
    datamodule = wrap_class(config["data"]["datamodule"])
    dm = datamodule()
    dm.setup("validate")

    # validation
    config["training"]["gpus"] = 1
    config["training"]["logger"] = False
    trainer = Trainer(**config["training"])
    trainer.validate(
        model=model, val_dataloaders=dm.val_dataloader(), ckpt_path=None
    )
    results = move_to(model.val_results, device="cpu")

    # convert to dataframe
    results = pd.DataFrame(results)
    results = results.replace([np.inf, -np.inf], np.nan)
    # results = pd.concat((results, results.describe()))

    # save results
    results.to_csv(config["test"]["save_path"])


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="""Evaluate on validation data.""", add_help=False
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


# class Arg(object):
#     config = os.path.join(
#         os.getcwd(), "configs/ablations/config_brats_fold0_senmf_iters2.py",
#     )


if __name__ == "__main__":
    args = get_args()
    # args = Arg()
    main(args)

