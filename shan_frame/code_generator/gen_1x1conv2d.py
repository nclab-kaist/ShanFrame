import numpy as np
import math
from ..ir.operator import Conv2D
from ..ir import DataLayout, Tensor, Model
from .output_code import KernelFunc, OutputCode, VecMulFunc
from .utils import (
    buffer_name,
    effective_scale,
    get_contribution,
    indent_lines,
    kernel_name,
    scales_name,
    contrib_name,
    weight_name,
)
from .code_pieces import *


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


def vec_mul_high_to_low(start: int, num: int, param: VecMulParam, indent: int) -> str:
    result = ""
    if num > 1:
        result += conv2d_1x1_even_loop_high_to_low(start, num//2, param.out_c, param.o2func,
                                                   param.weight, param.scales, param.contrib, param.out_offset, indent)
    if num % 2 != 0:
        result += conv2d_1x1_odd_cleanup(param.out_c, param.o1func,
                                         param.weight, param.scales, param.contrib, param.out_offset, indent)
    return result


def vec_mul_low_to_high(start: int, num: int, param: VecMulParam, indent: int) -> str:
    result = ""
    if num > 1:
        result += conv2d_1x1_even_loop_low_to_high(start, num//2, param.out_c, param.o2func,
                                                   param.weight, param.scales, param.contrib, param.out_offset, indent)
    if num % 2 != 0:
        result += indent_lines("input_elem -= input_c;\n", indent)
        result += conv2d_1x1_odd_cleanup(param.out_c, param.o1func,
                                         param.weight, param.scales, param.contrib, param.out_offset, indent)
    return result


def generate_1x1conv2d_naive_content(op: Conv2D, model: Model, output_code: OutputCode) -> str:
    input = model.tensors[op.input_idx]
    output = model.tensors[op.output_idx]
    weight = model.tensors[op.weight_idx]

    weight_str = weight_name(op.idx)
    scales_str = scales_name(op.idx)
    contrib_str = contrib_name(op.idx)
    out_offset = int(output.zero_point[0])
    indent = 1
    content = ""
    o2_vec_mul = output_code.add_vec_mul(2, input.dim_c, output.layout)
    o1_vec_mul = output_code.add_vec_mul(1, input.dim_c, output.layout)
    content += conv2d_1x1_setup(input.dim_h *
                                input.dim_w, output.layout, indent)

    sub_op_num = output.dim_h * output.dim_w

    # determine exec order
    input_floor = input.addr
    input_ceiling = input.addr + input.mem_size()
    input_size = input_ceiling - input_floor
    output_floor = output.addr
    output_ceiling = output.addr + output.mem_size()
    param = VecMulParam(o2_vec_mul, o1_vec_mul, output.dim_c,
                        weight_str, scales_str, contrib_str, str(out_offset))

    is_overlap = input_floor >= output_ceiling or output_floor >= input_ceiling
    low_to_high = vec_mul_low_to_high(0, sub_op_num, param, indent)
    high_to_low = vec_mul_high_to_low(input_size, sub_op_num, param, indent)
    if not is_overlap:
        # no overlapping
        content = low_to_high
    else:
        assert input.layout == output.layout == DataLayout.HWC, "conv2d io overlapping only support hwc"
        # check if linear exec is sufficient
        if input_floor <= output_floor and input_ceiling <= output_ceiling:
            # input is completely lower than output, low to high exec
            content = low_to_high
        elif input_floor >= output_floor and input_ceiling >= output_ceiling:
            # input is completely higher than output, high to low exec
            content = high_to_low
        elif input_ceiling <= output_ceiling and input_floor >= output_floor:
            # output contains input
            low_to_high_num = (
                input_floor - output_floor) // (output.dim_c - input.dim_c)
            content += vec_mul_low_to_high(0, low_to_high_num, param, indent)
            high_to_low_num = sub_op_num - low_to_high_num
            content += vec_mul_high_to_low(input_size,
                                           high_to_low_num, param, indent)
        elif input_ceiling >= output_ceiling and input_floor <= output_floor:
            # input contains output
            low_to_high_num = math.ceil(
                (input_ceiling - output_ceiling) / (input.dim_c - output.dim_c))
            start = input_size - input.dim_c * low_to_high_num
            content += vec_mul_low_to_high(start,
                                           low_to_high_num, param, indent)
            high_to_low_num = sub_op_num - low_to_high_num
            content += vec_mul_high_to_low(start,
                                           high_to_low_num, param, indent)
        else:
            raise NotImplementedError()
    # TODO: change out accordingly

    return content


# def generate_1x1conv2d_opad(op: Conv2D, model: Model, output_code: OutputCode) -> str:


def generate_1x1conv2d(op: Conv2D, model: Model, output_code: OutputCode) -> None:
    input = model.tensors[op.input_idx]
    output = model.tensors[op.output_idx]
    weight = model.tensors[op.weight_idx]

    bias = model.tensors[op.bias_idx].data
    contribution = get_contribution(weight, bias, input.zero_point[0])
    scales = effective_scale(input.scales, weight.scales, output.scales)

    assert weight.dim_h == weight.dim_w == 1
    input_pad = not op.pad_h == op.pad_w == 0
    output_pad = not output.prepad_h == output.prepad_w == 0

    content = ""
    match input_pad, output_pad:
        case False, False: content = generate_1x1conv2d_naive_content(op, model, output_code)
        case _: raise NotImplementedError()

    func = KernelFunc()
    func.content = content
    func.const = [
        (weight_name(op.idx), weight.data),
        (contrib_name(op.idx), contribution),
        (scales_name(op.idx), scales)
    ]
    func_name = kernel_name(op.idx, "conv2d_1x1")
    input_addr = f"&{buffer_name()}[{input.addr}]"
    output_addr = f"&{buffer_name()}[{output.addr}]"
    func.call = f"{func_name}({input_addr}, {output_addr})"
    func.definition = f"void {func_name}(int8_t *input, int8_t *output)"

    output_code.kernels[op.idx] = func
