from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, seed_everything

from registry import read_config


seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
print("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # get config
    config = read_config(args.config)

    # setup data
    dm = config["data"]
    dm.setup("test")

    # load model
    task_cls, task_params = config["task"]
    if (
        "checkpoint" in config["test"]
        and "checkpoint_path" in config["test"]["checkpoint"]
    ):
        model = task_cls.load_from_checkpoint(
            **config["test"]["checkpoint"], strict=False, **task_params
        )
    else:
        model = task_cls(**task_params)

    # validation
    config["training"]["logger"] = False
    trainer = Trainer(**config["training"])
    trainer.validate(model=model, dataloaders=dm.test_dataloader())


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="""Evaluate on validation data.""", add_help=False
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
