from numpy import ndarray
from ..ir import Tensor

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
    
    def __init__(self, col_num: int, col_size: int) -> None:
        self.col_num = col_num
        self.col_size = col_size
        
    def get_name(self) -> str:
        return f"vec_mul_1x{self.col_size}_{self.col_num}"
    
    def get_def(self) -> str:
        ret = "int8_t *"
        args = "const int8_t *input, int8_t *output, const int8_t *weight, int row_count, const float *scales, const int32_t contrib"
        return f"{ret}{self.get_name}({args})"
    
    def get_call(self, input: str, output: str, weight: str, row_count: str, scales: str, contrib: str) -> str:
        return f"{self.get_name}({input}, {output}, {weight}, {row_count}, {scales}, {contrib})"
    
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
    
    def add_vec_mul(self, col_num: int, col_size: int) -> VecMulFunc:
        vec_mul = self.vec_mul.get((col_num, col_size))
        if vec_mul is not None:
            return vec_mul
        vec_mul = self.vec_mul[(col_num, col_size)] = VecMulFunc(col_num, col_size)
        if col_num > 1:
            self.add_vec_mul(col_num - 1, col_size)
        return vec_mul
    