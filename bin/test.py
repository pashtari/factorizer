from argparse import ArgumentParser, Namespace

import torch
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
    dm.setup("test")

    # test
    config["training"]["gpus"] = 1
    config["training"]["logger"] = False
    trainer = Trainer(**config["training"])
    trainer.test(model=model, test_dataloaders=dm.test_dataloader())


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="""Do inference on test data.""", add_help=False
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


# class Arg(object):
#     config = [
#         os.path.join(os.getcwd(), "configs/config_msseg_preact-unet_fold0.py")
#     ]


if __name__ == "__main__":
    args = get_args()
    # args = Arg()
    main(args)

