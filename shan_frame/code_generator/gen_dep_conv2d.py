import numpy as np
from ..ir.operator import DepthConv2D
from .output_code import ChConvFunc, OutputCode, KernelFunc
from .dep_conv2d_code_pieces import *
from .utils import indent_lines

def generate_content(model: Model, op: DepthConv2D, output_code: OutputCode) -> str:
    input = model.tensors[op.input_idx]
    weight = model.tensors[op.weight_idx]
    output = model.tensors[op.output_idx]
    
    input_ceiling = input.addr + input.mem_size()
    output_ceiling = output.addr + output.mem_size()
    if not (input_ceiling <= output.addr or input.addr >= output_ceiling):
        # input output overlapped
        assert output.addr <= input.addr and (input.addr - output.addr) % input.dim_c == 0
    
    pad_input = op.pad_h != 0 or op.pad_w != 0
    pad_output = output.prepad_h != 0 or output.prepad_w != 0
    
    indent = 1
    content = ""
    content += depconv_setup(model, op, indent)
    if pad_input:
        content += depconv_pad_buffer(op.pad_h, op.pad_w, indent)
    content += depconv_loop_setup(indent)
    indent += 1    
    content += depconv_loop_body(input, weight, op, output_code, indent)
    if pad_output:
        raise NotImplementedError("prepad output for depthwise conv2d")
    indent -= 1
    content += indent_lines("}", indent)
    return content
    

def generate_depthwise_conv2d(model: Model, op: DepthConv2D, output_code: OutputCode) -> None:
    bias = model.tensors[op.bias_idx]
    input = model.tensors[op.input_idx]
    weight = model.tensors[op.weight_idx]
    output = model.tensors[op.output_idx]
    contribution = get_depthwise_conv2d_contribution(weight, bias, input.zero_point[0])
    scales = effective_scale(input.scales, weight.scales, output.scales)
    
    hwc_to_chw(weight)
    func_name = f"layer{op.idx}_depthwise_conv2d"
    input_addr = f"{buffer_name()}[{input.addr}]"
    output_addr = f"{buffer_name()}[{output.addr}]"
    buffer_addr = f"{buffer_name()}[{op.buffer_addr}]"
    
    func = KernelFunc()
    func.call = f"{func_name}({input_addr}, {output_addr}, {buffer_addr})"
    func.definition = f"void {func_name}(const int8_t *input, int8_t *output, int8_t *buffer)"
    func.content = generate_content(model, op, output_code)
    func.const = [
        (weight_name(op.idx), weight.data),
        (contrib_name(op.idx), contribution),
        (scales_name(op.idx), scales)
    ]

    