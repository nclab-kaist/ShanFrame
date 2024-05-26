from numpy import ndarray
from ..ir import DataLayout, Tensor

class KernelFunc:
    name: str
    param: str
    include: list[str]
    const: list[tuple[str, ndarray]]
    content: list[str]
    current_indent: int
    
    def __init__(self, name: str, param: str) -> None:
        self.name = name
        self.param = param
        self.include = []
        self.const = []
        self.content = []
        self.current_indent = 1
    
    def add_line(self, line: str) -> None:
        indent_str = "    " * self.current_indent
        self.content.append(indent_str + line)
        
    def add_include(self, header: str) -> None:
        self.include.append(header)
    
    def add_const(self, name: str, data: ndarray) -> None:
        self.const.append((name, data))
        
    def add_indent(self) -> None:
        self.current_indent += 1
        
    def del_indent(self) -> None:
        self.current_indent -= 1

class VecMulFunc:
    col_num: int
    col_size: int
    output_layout: DataLayout
    
    def __init__(self, col_num: int, col_size: int, output_layout: DataLayout) -> None:
        self.col_num = col_num
        self.col_size = col_size
        self.output_layout = output_layout
        
    def get_name(self) -> str:
        return f"vec_mul_1x{self.col_size}_{self.col_num}_{self.output_layout}"
    
    def get_def(self) -> str:
        ret = "int8_t *"
        name = self.get_name()
        args = "const int8_t *input, int8_t *output, const int8_t *weight, const int row_count, const int ch_offset, const int out_offset, const float *scales, const int32_t *contrib"
        return f"{ret}{name}({args})"
    
    def get_call(self, input: str, output: str, weight: str, row_count: str, ch_offset: str, output_offset: str, scales: str, contrib: str) -> str:
        return f"{self.get_name()}({input}, {output}, {weight}, {row_count}, {ch_offset}, {output_offset}, {scales}, {contrib})"
    
class OutputCode:
    root_dir: str
    kernels: dict[str, KernelFunc]
    vec_mul: dict[tuple[int, int], VecMulFunc]
    const_tensors: list[Tensor]
    
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.kernels = {}
        self.vec_mul = {}
        self.const_tensors = []
    
    def add_vec_mul(self, col_num: int, col_size: int, output_layout: DataLayout) -> VecMulFunc:
        vec_mul = self.vec_mul.get((col_num, col_size))
        if vec_mul is not None:
            return vec_mul
        vec_mul = self.vec_mul[(col_num, col_size)] = VecMulFunc(
            col_num, col_size, output_layout)
        return vec_mul
    