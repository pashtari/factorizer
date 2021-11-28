from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, seed_everything

from registry import read_config


seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
print("cuda" if torch.cuda.is_available() else "cpu")


def main(args: Namespace):
    # get config
    config = read_config(args.config)

    # data
    dm = config["data"]

    # init model
    task_cls, task_params = config["task"]
    if "checkpoint_path" in task_params:
        model = task_cls.load_from_checkpoint(strict=False, **task_params)
    else:
        model = task_cls(**task_params)

    # init trainer
    trainer = Trainer(**config["training"])

    # fit model
    trainer.fit(model, dm)


def get_args() -> Namespace:
    parser = ArgumentParser(description="""Train the model.""", add_help=False)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
