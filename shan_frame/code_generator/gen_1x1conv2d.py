import numpy as np
import math
from ..ir.operator import Conv2D
from ..ir import DataLayout, Tensor, Model
from .output_code import KernelFunc, OutputCode, VecMulFunc
from .utils import (
    buffer_name,
    effective_scale,
    get_conv2d_contribution,
    indent_lines,
    kernel_name,
    scales_name,
    contrib_name,
    weight_name,
)
from .conv2d_code_pieces import *


class VecMulParam:
    o2func: VecMulFunc
    o1func: VecMulFunc
    out_c: int
    weight: str
    scales: str
    contrib: str
    out_offset: str

    def __init__(self, o2func: VecMulFunc, o1func: VecMulFunc, out_c: int, weight: str, scales: str, contrib: str, out_offset: str) -> None:
        self.o2func = o2func
        self.o1func = o1func
        self.out_c = out_c
        self.weight = weight
        self.scales = scales
        self.contrib = contrib
        self.out_offset = out_offset


def gen_non_overlap_content(idx: int, input: Tensor, output: Tensor, output_code: OutputCode) -> str:
    indent = 1
    content = conv2d_setup(idx, input, output, indent)
    # prepad output
    content += conv2d_prepad(output, indent)
    out_start = output.prepad_h * \
        (output.dim_w + 2 * output.prepad_w) + output.prepad_w
    if output.prepad_w == 0:
        content += conv2d_1x1_all(input, output, output_code, indent)
    else:
        content += conv2d_1x1_by_row(input, output, output_code, indent)

    return content


def gen_overlap_content(idx: int, input: Tensor, output: Tensor, output_code: OutputCode) -> str:
    indent = 1
    content = conv2d_setup(idx, input, output, indent)
    assert output.prepad_h == output.prepad_w == 0
    assert input.layout == output.layout == DataLayout.HWC

    input_floor = input.addr
    input_ceiling = input.addr + input.mem_size()
    output_floor = output.addr
    output_ceiling = output.addr + output.mem_size()
    
    input_start1 = 0
    out_start1 = 0
    l_h_num = 0
    input_start2 = 0
    out_start2 = 0
    h_l_num = 0
    total_num = output.dim_h * output.dim_w
    
    if input_floor <= output_floor and input_ceiling <= output_ceiling:
        # input is completely lower than output, high to low exec
        input_start2 = input.mem_size()
        out_start2 = output.mem_size()
        h_l_num = total_num
    elif input_floor >= output_floor and input_ceiling >= output_ceiling:
        # input is completely higher than output, high to low exec
        l_h_num = total_num
    elif input_ceiling <= output_ceiling and input_floor >= output_floor:
        # output contains input
        l_h_num = (
            input_floor - output_floor) // (output.dim_c - input.dim_c)
        h_l_num = total_num - l_h_num
        input_start2 = l_h_num * input.dim_c
        out_start2 = l_h_num * output.dim_c
    elif input_ceiling >= output_ceiling and input_floor <= output_floor:
        # input contains output
        l_h_num = math.ceil(
            (input_ceiling - output_ceiling) / (input.dim_c - output.dim_c))
        input_start1 = input.mem_size() - input.dim_c * l_h_num
        out_start1 = output.mem_size() - output.dim_c * l_h_num
        h_l_num = total_num - l_h_num
        input_start2 = input_start1
        out_start2 = out_start1
    else:
        raise NotImplementedError(f"input: {input_floor}~{input_ceiling}, output: {output_floor}~{output_ceiling}")
    
    content += conv2d_1x1_low_to_high(input_start1, out_start1, l_h_num, input, output, output_code, indent)
    content += conv2d_1x1_high_to_low(input_start2, out_start2, h_l_num, input, output, output_code, indent)
    return content
    

def generate_1x1conv2d(op: Conv2D, model: Model, output_code: OutputCode) -> None:
    input = model.tensors[op.input_idx]
    output = model.tensors[op.output_idx]
    weight = model.tensors[op.weight_idx]
    bias = model.tensors[op.bias_idx]
    contribution = get_conv2d_contribution(weight, bias, input.zero_point[0])
    scales = effective_scale(input.scales, weight.scales, output.scales)
    
    assert op.stride_h == op.stride_w == 1
    
    input_floor = input.addr
    input_ceiling = input.addr + input.mem_size()
    output_floor = output.addr
    output_ceiling = output.addr + output.mem_size()
    is_overlap = not (
        input_floor >= output_ceiling or output_floor >= input_ceiling)

    assert weight.dim_h == weight.dim_w == 1
    input_pad = not op.pad_h == op.pad_w == 0
    output_pad = not output.prepad_h == output.prepad_w == 0

    content = ""
    match is_overlap, input_pad, output_pad:
        case _, True, _: raise NotImplementedError("1x1 conv2d input padding")
        case False, False, _: content = gen_non_overlap_content(op.idx, input, output, output_code)
        case True, _, True: raise NotImplementedError("overlapped output pre-padding")
        case True, False, False: content = gen_overlap_content(op.idx, input, output, output_code)

    func = KernelFunc()
    func.content = content
    func.const = [
        (f"const int8_t {weight_name(op.idx)}[]", weight.data),
        (f"const int32_t {contrib_name(op.idx)}[]", contribution),
        (f"const float {scales_name(op.idx)}[]", scales)
    ]
    func_name = kernel_name(op.idx, "conv2d_1x1")
    input_addr = f"&{buffer_name()}[{input.addr}]"
    output_addr = f"&{buffer_name()}[{output.addr}]"
    func.call = f"{func_name}({input_addr}, {output_addr})"
    func.definition = f"void {func_name}(int8_t *input, int8_t *output)"

    output_code.kernels[op.idx] = func
