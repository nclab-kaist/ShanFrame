import numpy as np
from ..ir import Operator, Model, Tensor
from ..ir.operator import *


def gen_copy_int8(src: str, dst: str, size: str) -> list[str]:
    # XXX: rely on memcpy, may not be most efficient
    return [f"memcpy({dst}, {src}, {size})"]

def concat_line(prev: str, next: str, indent: int) -> str:
    indent_str = "    "
    return f"{prev}{indent_str * indent}{next}\n"

def indent_lines(input: str, indent: int) -> str:
    indent_str = "    "
    lines = [line.strip() for line in input.splitlines()]
    lines = list(filter(lambda line: len(line) > 0, lines))
    lines = [indent_str * indent + line + "\n" for line in lines]
    return "".join(lines)

def effective_scale(input_scales: np.ndarray, weight_scales: np.ndarray, output_scales: np.ndarray) -> np.ndarray:
    return input_scales * weight_scales / output_scales


def get_conv2d_contribution(weight: Tensor, bias: Tensor, input_zero: int) -> np.ndarray:
    weight_shaped = np.reshape(
        weight.data, (weight.dim_n, weight.dim_h, weight.dim_w, weight.dim_c))
    weight_sum = np.sum(weight_shaped, axis=(1, 2, 3))
    return bias.data + weight_sum * -input_zero


def get_depthwise_conv2d_contribution(weight: Tensor, bias: Tensor, input_zero: int) -> np.ndarray:
    weight_shaped = np.reshape(
        weight.data, (weight.dim_n, weight.dim_h, weight.dim_w, weight.dim_c))
    weight_sum = np.sum(weight_shaped, axis=(0, 1, 2))
    return bias.data + weight_sum * -input_zero

def scales_name(idx: int) -> str:
    return f"scales{idx}"


def scales_declare(idx: int) -> str:
    return f"const float {scales_name(idx)}[]"


def contrib_name(idx: int) -> str:
    return f"contrib{idx}"


def contrib_declare(idx: int) -> str:
    return f"const int32_t {contrib_name(idx)}[]"


def weight_name(idx: int) -> str:
    return f"weight{idx}"


def weight_declare(idx: int) -> str:
    return f"const int8_t {weight_name(idx)}[]"


def buffer_name() -> str:
    return "buffer"


def kernel_name(idx: int, op_type: str) -> str:
    return f"layer{idx}_{op_type}"

def hwc_to_chw(tensor: Tensor):
    assert tensor.layout == DataLayout.HWC
    tensor.data = np.reshape(tensor.data, (tensor.dim_n, tensor.dim_h, tensor.dim_w, tensor.dim_c))
    tensor.data = np.transpose(tensor.data, (0, 3, 1, 2))
    tensor.layout = DataLayout.CHW
    