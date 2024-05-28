import numpy as np

from .utils import effective_scale, get_contribution
from ..ir.operator import Conv2D
from ..ir import Tensor, Model
from .output_code import OutputCode, KernelFunc
from .gen_1x1conv2d import generate_1x1conv2d
from .code_pieces import *


def gen_content(model: Model, op: Conv2D, output_code: OutputCode) -> str:
    input = model.tensors[op.input_idx]
    weight = model.tensors[op.weight_idx]
    output = model.tensors[op.output_idx]

    indent = 1
    content = conv2d_setup(op.idx, input, output, indent)
    content += conv2d_prepad(output, indent)    
    content += conv2d_window_slide(op, input, weight, output, output_code, indent)
    return content


def generate_conv2d(model: Model, op: Conv2D, output_code: OutputCode):
    weight = model.tensors[op.weight_idx]
    if weight.dim_h == weight.dim_w == 1:
        return generate_1x1conv2d(op, model, output_code)
    input = model.tensors[op.input_idx]
    output = model.tensors[op.output_idx]
    if input.data.dtype != np.int8:
        raise NotImplementedError(
            f"ShanFrame does not support {input.data.dtype}")

    bias = model.tensors[op.bias_idx]
    input = model.tensors[op.input_idx]
    weight = model.tensors[op.weight_idx]
    output = model.tensors[op.output_idx]
    scales = effective_scale(input.scales, weight.scales, output.scales)
    contribution = get_contribution(weight, bias, input.zero_point[0])
    
    func = KernelFunc()
    func.content = gen_content(model, op, output_code)
    func.const = [
        (weight_name(op.idx), weight.data),
        (contrib_name(op.idx), contribution),
        (scales_name(op.idx), scales)
    ]
    func_name = kernel_name(op.idx, "conv2d")
    input_addr = f"&{buffer_name()}[{input.addr}]"
    output_addr = f"&{buffer_name()}[{output.addr}]"
    buffer_addr = f"&{buffer_name()}[{op.buffer_addr}]"
    func.call = f"{func_name}({input_addr}, {output_addr}, {buffer_addr})"
    func.definition = f"void {func_name}(const int8_t *input, int8_t *output, int8_t *buffer)"
    
    output_code.kernels[op.idx] = func

