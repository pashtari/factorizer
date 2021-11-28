# %%
from argparse import ArgumentParser, Namespace
import os
from importlib import import_module
import sys
import json
import re

import torch
from torch.utils import benchmark
from pytorch_lightning import seed_everything

from factorizer.utils.lightning import Model


seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda")
    cuda = True
else:
    print("cpu")
    device = torch.device("cpu")
    cuda = False

# os.chdir(os.path.dirname(__file__))

# %%
def main(args):
    fuzzer = benchmark.Fuzzer(
        parameters=[],
        tensors=[
            benchmark.FuzzedTensor("x", size=(1, 4, 128, 128, 128), cuda=cuda)
        ],
        seed=42,
    )

    results = []
    for config_path in args.config:
        # get config
        directory = os.path.dirname(config_path)
        sys.path.insert(0, directory)
        base = os.path.basename(config_path)
        file_name = os.path.splitext(base)[0]
        module = import_module(file_name, directory)
        config = getattr(module, "CONFIG")

        # load model
        model = Model.load_from_checkpoint(
            **config["test"]["checkpoint"],
            strict=False,
            **config["model"],
            **config["optimization"],
        )
        model.freeze()
        model.to(device)

        # benchmark inference time
        measures = []
        for tensors, _, _ in fuzzer.take(2):
            bench = benchmark.Timer(
                "model(x)",
                globals={"model": model, **tensors},
                label=file_name,
                description="inference time",
            ).blocked_autorange(min_run_time=1)
            measures.append(bench)

        measures = benchmark.Measurement.merge(measures)[0]
        results.append(measures)

        # save results
        path = config["test"]["save_path"]
        path = re.sub("results", "benchmarks", path)
        path = re.sub(".csv", ".json", path)

        measures._lazy_init()
        with open(path, "w") as fp:
            json.dump(measures, fp, default=lambda o: o.__dict__)

    # aggregate results
    compare = benchmark.Compare(results)
    compare.colorize()
    compare.print()


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="""Calculate inference time.""", add_help=False
    )
    parser.add_argument("--config", nargs="+", required=True)
    args = parser.parse_args()
    return args


# class Arg(object):
#     config = [
#         os.path.join(os.getcwd(), "configs/config_brats_unet_preact.py"),
#         os.path.join(os.getcwd(), "configs/config_brats_unet_basic.py"),
#     ]


if __name__ == "__main__":
    args = get_args()
    # args = Arg()
    main(args)
