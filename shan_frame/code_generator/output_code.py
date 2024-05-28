from numpy import ndarray
from ..ir import DataLayout, Tensor

class KernelFunc:
    include: str
    definition: str
    const: list[tuple[str, ndarray]]
    content: str
    call: str
    
    def __init__(self) -> None:
        self.include = ""
        self.definition = ""
        self.const = []
        self.content = ""
        
    def print_def(self) -> str:
        ret = self.definition
        ret += "{\n"
        ret += self.content
        ret += "}\n"
        return ret

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
        ret = "void "
        name = self.get_name()
        args = "const int8_t *input, int8_t *output, const int8_t *weight, const int row_count, const int ch_offset, const int out_offset, const float *scales, const int32_t *contrib"
        return f"{ret}{name}({args})"
    
    def get_call(self, input: str, output: str, weight: str, row_count: str, ch_offset: str, output_offset: str, scales: str, contrib: str) -> str:
        return f"{self.get_name()}({input}, {output}, {weight}, {row_count}, {ch_offset}, {output_offset}, {scales}, {contrib})"
    
class OutputCode:
    root_dir: str
    kernels: dict[int, KernelFunc]
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
    