"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from torch.nn.utils.weight_norm import remove_weight_norm, WeightNorm

fc_default_activation = th.nn.LeakyReLU(0.2, inplace=True)


class FCLayer(th.nn.Module):
    def __init__(self, n_in, n_out, nonlin=fc_default_activation) -> None:
        super().__init__()
        self.fc = th.nn.Linear(n_in, n_out, bias=True)
        self.nonlin = nonlin if nonlin is not None else lambda x: x

        self.fc.bias.data.fill_(0)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        x = self.fc(x)
        x = self.nonlin(x)
        return x


def check_args_shadowing(name, method: object, arg_names) -> None:
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(f"{name} attempted to shadow a wrapped argument: {arg_name}")


class TensorMappingHook:
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.

        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.

            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor


def weight_norm_wrapper(
    cls: Type[th.nn.Module],
    new_cls_name: str,
    name: str = "weight",
    g_dim: int = 0,
    v_dim: Optional[int] = 0,
):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.

    Args:
        cls: Type[th.nn.Module]
        Class to apply the wrapper to.

        new_cls_name: str
        Name of the new class created by the wrapper. This should be the name
        of whatever variable you assign the result of this function to. Ex:
        ``SomeLayerWN = weight_norm_wrapper(SomeLayer, "SomeLayerWN", ...)``

        name: str
        Name of the parameter to apply weight normalization to.

        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.

        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(self, *args: Any, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs: Any):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = th.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    Wrap.__qualname__ = new_cls_name

    return Wrap


def is_weight_norm_wrapped(module) -> bool:
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False


class Conv2dUB(th.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        *args,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """Conv2d with untied bias."""
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    def _conv_forward(self, input: th.Tensor, weight: th.Tensor, bias: Optional[th.Tensor]):
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return F.conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: th.Tensor) -> th.Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class ConvTranspose2dUB(th.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        *args,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """ConvTranspose2d with untied bias."""
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)

        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose2dUB")

        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    def forward(self, input: th.Tensor, output_size: Optional[List[int]] = None) -> th.Tensor:
        output_padding = self._output_padding(
            input=input,
            output_size=output_size,
            stride=self.stride,
            padding=self.padding,
            kernel_size=self.kernel_size,
            num_spatial_dims=input.dim() - 2,
            dilation=self.dilation,
        )

        output = F.conv_transpose2d(
            input,
            self.weight,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output

    def _output_padding(
        self,
        input: th.Tensor,
        output_size: Optional[List[int]],
        stride: List[int],
        padding: List[int],
        kernel_size: List[int],
        num_spatial_dims: int,
        dilation: Optional[List[int]] = None,
    ) -> List[int]:
        if output_size is None:
            # converting to list if was not already
            ret = th.nn.modules.utils._single(self.output_padding)
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = th.jit.annotate(List[int], [])
            max_sizes = th.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = th.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


# Set default g_dim=0 (Conv2d) or 1 (ConvTranspose2d) and v_dim=None to preserve
# the current weight norm behavior.
LinearWN = weight_norm_wrapper(th.nn.Linear, "LinearWN", g_dim=0, v_dim=None)
Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, "Conv2dWN", g_dim=0, v_dim=None)
Conv2dWNUB = weight_norm_wrapper(Conv2dUB, "Conv2dWNUB", g_dim=0, v_dim=None)
ConvTranspose2dWN = weight_norm_wrapper(
    th.nn.ConvTranspose2d, "ConvTranspose2dWN", g_dim=1, v_dim=None
)
ConvTranspose2dWNUB = weight_norm_wrapper(
    ConvTranspose2dUB, "ConvTranspose2dWNUB", g_dim=1, v_dim=None
)


class InterpolateHook:
    def __init__(self, size=None, scale_factor=None, mode: str = "bilinear") -> None:
        """An object storing options for interpolate function"""
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, module, x):
        assert len(x) == 1, "Module should take only one input for the forward method."
        return F.interpolate(
            x[0],
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
        )


def interpolate_wrapper(cls: Type[th.nn.Module], new_cls_name: str):
    """Wraps a torch.nn.Module class and perform additional interpolation on the
    first and only positional input of the forward method.

    Args:
        cls: Type[th.nn.Module]
        Class to apply the wrapper to.

        new_cls_name: str
        Name of the new class created by the wrapper. This should be the name
        of whatever variable you assign the result of this function to. Ex:
        ``UpConv = interpolate_wrapper(Conv, "UpConv", ...)``

    """

    class Wrap(cls):
        def __init__(
            self, *args: Any, size=None, scale_factor=None, mode="bilinear", **kwargs: Any
        ):
            check_args_shadowing(
                "interpolate_wrapper", super().__init__, ["size", "scale_factor", "mode"]
            )
            super().__init__(*args, **kwargs)
            self.register_forward_pre_hook(
                InterpolateHook(size=size, scale_factor=scale_factor, mode=mode)
            )

    Wrap.__qualname__ = new_cls_name
    return Wrap


UpConv2d = interpolate_wrapper(th.nn.Conv2d, "UpConv2d")
UpConv2dWN = interpolate_wrapper(Conv2dWN, "UpConv2dWN")
UpConv2dWNUB = interpolate_wrapper(Conv2dWNUB, "UpConv2dWNUB")


def glorot(m: th.nn.Module, alpha: float = 1.0) -> None:
    gain = np.sqrt(2.0 / (1.0 + alpha**2))

    if isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()


def make_tuple(x: Union[int, Tuple[int, int]], n: int) -> Tuple[int, int]:
    if isinstance(x, int):
        return tuple([x for _ in range(n)])
    else:
        return x
