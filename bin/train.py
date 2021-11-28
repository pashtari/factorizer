from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer, seed_everything

from factorizer.utils.helpers import wrap_class, read_config
from factorizer.utils.lightning import Model


seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
print("cuda" if torch.cuda.is_available() else "cpu")

# os.chdir(os.path.dirname(__file__))


def main(args: Namespace):
    # get config
    config = read_config(args.config)

    # init model
    if "checkpoint_path" in config["model"]:
        model = Model.load_from_checkpoint(
            strict=False, **config["model"], **config["optimization"]
        )
    else:
        model = Model(**config["model"], **config["optimization"])

    # init data
    datamodule = wrap_class(config["data"]["datamodule"])
    dm = datamodule()

    from torch import autograd

    with autograd.detect_anomaly():
        x = torch.rand(1, 4, 128, 128, 128)
        y = model(x)
        print(f"shape: {y[0].shape}")

        loss = y[0].sum()
        loss.backward()

    # init trainer
    trainer = Trainer(**config["training"])

    # fit model
    trainer.fit(model, dm)


def get_args() -> Namespace:
    parser = ArgumentParser(description="""Train the model.""", add_help=False)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


class Arg(object):
    config = "configs/ablations/config_brats_fold0_lnmf-hals.py"


if __name__ == "__main__":
    # args = get_args()  # for running from terminal
    args = Arg()
    main(args)

