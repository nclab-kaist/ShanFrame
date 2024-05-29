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
        args = ", ".join([
            "const int8_t *input", "int8_t *output", "const int8_t *weight", 
            "const int row_count", "const int ch_offset", "const int out_offset", 
            "const float *scales", "const int32_t *contrib"
        ])
        return f"{ret}{name}({args})"
    
    def get_call(self, 
                 input: str, output: str, weight: str, 
                 row_count: str, ch_offset: str, output_offset: str, 
                 scales: str, contrib: str) -> str:
        return f"{self.get_name()}({input}, {output}, {weight}, {row_count}, {ch_offset}, {output_offset}, {scales}, {contrib})"


class ChConvFunc:
    kernel_size: int
    stride: int
    rev: bool

    def __init__(self, kernel_size: int, stride: int, rev: bool) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.rev = rev

    def get_name(self) -> str:
        name = f"ch_conv_{self.kernel_size}x{self.kernel_size}_stride{self.stride}"
        if self.rev:
            name += "_rev"
        return name

    def get_def(self) -> str:
        ret = "void "
        name = self.get_name()
        args = ", ".join([
            "const int8_t *input", "int8_t *output", "const int8_t *ksrc",
            "const float scale", "const int32_t contrib", "const int8_t out_offset",
            "const int row_size", "const int ch_offset", "const int out_w", "const int out_h"
        ])
        return f"{ret}{name}({args})"
    
    def get_call(self,
                 input: str, output: str, ksrc: str,
                 scale: str, contrib: str, out_offset: str,
                 row_size: str, ch_offset: str, out_w: str, out_h: str) -> str:
        return f"{self.get_name()}({input}, {output}, {ksrc}, {scale}, {contrib}, {out_offset}, {row_size}, {ch_offset}, {out_w}, {out_h})"


class OutputCode:
    root_dir: str
    kernels: dict[int, KernelFunc]
    vec_mul: dict[tuple[int, int], VecMulFunc]
    ch_conv: dict[tuple[int, int, bool], ChConvFunc]
    const_tensors: list[Tensor]
    
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.kernels = {}
        self.vec_mul = {}
        self.ch_conv = {}
        self.const_tensors = []
    
    def add_vec_mul(self, col_num: int, col_size: int, output_layout: DataLayout) -> VecMulFunc:
        vec_mul = self.vec_mul.get((col_num, col_size))
        if vec_mul is not None:
            return vec_mul
        vec_mul = self.vec_mul[(col_num, col_size)] = VecMulFunc(
            col_num, col_size, output_layout)
        return vec_mul
    
    def add_ch_conv(self, kernel_size: int, stride: int, rev: bool) -> ChConvFunc:
        ch_conv = self.ch_conv.get((kernel_size, stride, rev))
        if ch_conv is not None:
            return ch_conv
        ch_conv = self.ch_conv[(kernel_size, stride, rev)] = ChConvFunc(kernel_size, stride, rev)
        return ch_conv