from .utils import buffer_name
from ..ir import DataLayout, Model
from ..ir.operator import AvgPool2D
from .output_code import OutputCode, KernelFunc

def generate_avgpool(input_var: str, model: Model, op: AvgPool2D, output_code: OutputCode) -> None:
    input = model.tensors[op.input_idx]
    output = model.tensors[op.output_idx]
    
    assert input.layout == output.layout == DataLayout.HWC
    
    if input.addr >= 0:
        input_addr = f"&{buffer_name()}[{input.addr}]"
    else:
        input_addr = input_var
    output_addr = f"&{buffer_name()}[{output.addr}]"
    
    func_name = "avg_pooling"
    args = ", ".join([
        input_addr, str(input.dim_h), str(input.dim_w), str(input.dim_c),
        output_addr, str(output.dim_h), str(output.dim_w), str(output.dim_c),
        str(op.filter_h), str(op.filter_w)
    ])
    func = KernelFunc()
    func.call = f"{func_name}({args})"
    func.definition = ""
    func.content = ""
    
    output_code.kernels[op.idx] = func
    