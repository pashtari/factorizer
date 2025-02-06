import torch
from torch import Tensor
import torch.nn.functional as F
from torch.func import vmap


CONV = {d: getattr(F, f"conv{d}d") for d in range(1, 4)}


@vmap
def conv1(input: Tensor, weight: Tensor, **kwargs) -> Tensor:
    spatial_dims = input.ndim - 1
    input = input.unsqueeze(0)
    output = CONV[spatial_dims](input, weight, **kwargs).squeeze(0)
    return output


def conv2(input: Tensor, weight: Tensor, groups: int = 1, **kwargs) -> Tensor:
    batch_size = input.shape[0]
    spatial_dims = input.ndim - 2

    # Reshape input and weight for batch computation
    input_reshaped = input.reshape(1, batch_size * input.shape[1], *input.shape[2:])
    weight_reshaped = weight.reshape(
        batch_size * weight.shape[1], weight.shape[2], *weight.shape[3:]
    )

    # Adjust groups for batch computation
    groups = groups * batch_size

    # Perform convolution
    output = CONV[spatial_dims](input_reshaped, weight_reshaped, groups=groups, **kwargs)

    # Reshape output back to batch form
    output = output.reshape(batch_size, -1, *output.shape[2:])

    return output


# Define random input and weight
batch_size = 4
in_channels = 3
out_channels = 2
spatial_size = (32, 32, 32)  # Change for 1D or 3D testing
kernel_size = (3, 3, 3)  # Change accordingly

# Generate random tensors
input_tensor = torch.randn(batch_size, in_channels, *spatial_size)
weight_tensor = torch.randn(
    batch_size, out_channels, in_channels // 1, *kernel_size
)  # groups=1

# Compute outputs
output1 = conv1(input_tensor, weight_tensor, stride=1, padding=1, groups=1)
output2 = conv2(input_tensor, weight_tensor, stride=1, padding=1, groups=1)

# Check equivalence
if torch.allclose(output1, output2, atol=1e-9):
    print("conv1 and conv2 produce equivalent outputs ✅")
else:
    print("conv1 and conv2 outputs are different ❌")
    print("Max difference:", (output1 - output2).abs().max())


@vmap
def sconv1(input1: Tensor, input2: Tensor, **kwargs) -> Tensor:
    spatial_dims = input1.ndim - 1
    input1 = input1.unsqueeze(1)
    input2 = input2.unsqueeze(1)
    out = CONV[spatial_dims](input1, input2, **kwargs)
    return out
