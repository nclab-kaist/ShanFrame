from .utils import buffer_name
from ..ir import Model, Tensor
from ..ir.operator import Add
from .output_code import OutputCode, KernelFunc

def generate_add(model: Model, op: Add, output_code: OutputCode):
    input1 = model.tensors[op.input_idx_list[0]]
    input2 = model.tensors[op.input_idx_list[1]]
    output = model.tensors[op.output_idx]
    
    assert input1.mem_size() == input2.mem_size() == output.mem_size()
    assert input1.layout == input2.layout == output.layout
    
    input1_addr = f"&{buffer_name()}[{input1.addr}]"
    input2_addr = f"&{buffer_name()}[{input2.addr}]"
    output_addr = f"&{buffer_name()}[{output.addr}]"
    
    func = KernelFunc()
    func_name = "elementwise_add"
    args = ", ".join([
        f"{input1.mem_size()}", 
        input1_addr, str(input1.scales[0]), str(input1.zero_point[0]),
        input2_addr, str(input2.scales[0]), str(input2.zero_point[0]),
        output_addr, str(output.scales[0]), str(output.zero_point[0]),
    ])
    func.call = f"{func_name}({args})"
    func.definition = ""
    func.content = ""
    
    output_code.kernels[op.idx] = func
    