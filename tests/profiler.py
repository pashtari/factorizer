from itertools import product
import json

import torch
from torch import nn
from monai.networks.nets import DynUNet, SegResNet, UNETR, SwinUNETR
from monai.utils import UpsampleMode

from deepspeed.profiling.flops_profiler import get_model_profile, duration_to_string
from deepspeed.accelerator import get_accelerator

import factorizer as ft


def profile_model(model):
    input_shape = (1, 2, 128, 128, 128)
    with get_accelerator().device("cpu"):
        flops, macs, params = get_model_profile(
            model=model,
            input_shape=input_shape,
            args=None,
            kwargs=None,
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=1,
            warm_up=1,
            as_string=True,
            output_file=None,
            ignore_modules=None,
            mode="forward",
        )

    input_tensor = torch.ones(()).new_empty(
        (*input_shape,),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as prof:
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)

    latency = sum(evt.self_cpu_time_total for evt in prof.key_averages()) / 1e6
    latency = duration_to_string(latency / 5)

    return flops, macs, params, latency


# Define the models
models = {}
for groups, ratio, ks in product([1, 4, 8, -1], [1, 4, 8], [3, 5, 7]):
    if (groups * ratio) <= 32 and (groups * ratio) >= -1:
        key = f"Deconver (groups={groups}, ratio={ratio}, kernel_size={ks})"
        models[key] = ft.Deconver(
            in_channels=2,
            out_channels=1,
            spatial_dims=3,
            encoder_depth=[1, 1, 1, 1, 1],
            encoder_width=[32, 64, 128, 256, 512],
            strides=[1, 2, 2, 2, 2],
            decoder_depth=[1, 1, 1, 1],
            norm=nn.InstanceNorm3d,
            act=nn.ReLU,
            groups=groups,
            ratio=ratio,
            kernel_size=[ks, ks, ks],
            num_iters=1,
            mlp_ratio=3,
        )

models.update(
    {
        "Factorizer": ft.Factorizer(
            in_channels=2,
            out_channels=1,
            spatial_size=(128, 128, 128),
            encoder_depth=[1, 1, 1, 1, 1],
            encoder_width=[32, 64, 128, 256, 512],
            strides=[1, 2, 2, 2, 2],
            decoder_depth=[1, 1, 1, 1],
            norm=ft.LayerNorm,
            reshape=[ft.SWMatricize, {"head_dim": 8, "patch_size": 8}],
            act=nn.ReLU,
            factorize=ft.NMF,
            rank=1,
            num_iters=5,
            init="uniform",
            solver="hals",
            mlp_ratio=3,
        ),
        "SegResNet": SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            upsample_mode=UpsampleMode.DECONV,
        ),
        "DynUNet": DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=[32, 64, 128, 256, 512],
            norm_name="instance",
            act_name="LeakyReLU",
            res_block=False,
        ),
        "UNETR": UNETR(
            in_channels=2,
            out_channels=1,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            spatial_dims=3,
        ),
        "SwinUNETR V1": SwinUNETR(
            in_channels=2,
            out_channels=1,
            img_size=(128, 128, 128),
            feature_size=24,
            norm_name="instance",
            downsample="merging",
            normalize=True,
            spatial_dims=3,
            use_checkpoint=False,
            use_v2=False,
        ),
        "SwinUNETR V2": SwinUNETR(
            in_channels=2,
            out_channels=1,
            img_size=(128, 128, 128),
            feature_size=24,
            norm_name="instance",
            downsample="merging",
            normalize=True,
            spatial_dims=3,
            use_checkpoint=False,
            use_v2=True,
        ),
    }
)

results = []

for model_name, model in models.items():
    print(f"\nProfiling {model_name}...")
    flops, macs, params, latency = profile_model(model)
    results.append(
        {
            "model": model_name,
            "flops": flops,
            "macs": macs,
            "params": params,
            "latency": latency,
        }
    )

with open("tests/profiler.json", "w") as outfile:
    json.dump(results, outfile)
