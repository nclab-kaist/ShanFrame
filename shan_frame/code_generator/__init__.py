from .gen_vecmul import generate_vec_mul_def
from .gen_ch_conv import generate_ch_conv_def
from .output_code import OutputCode
from .gen_conv2d import generate_conv2d
from .gen_dep_conv2d import generate_depthwise_conv2d
from .gen_add import generate_add
from .gen_avgpool import generate_avgpool
from ..ir.operator import Conv2D, DepthConv2D, Add, AvgPool2D, Reshape
from ..ir import Model
from .gen_ch_conv import test
import os

class CodeGenerator:
    output_dir: str
    include_dir: str = "include"
    src_dir: str = "src"
    layer_file_name_base = "layer"
    vec_mul_file_name = "vec_mul"
    ch_conv_file_name = "ch_conv"
    minor_op_file_name = "minor_op"
    kernel_file_name = "kernels"
    intrin_file_name = "intrinsics"
    const_file_name = "model_const"
    
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        
    def write_src(self, name: str, lines: list[str]) -> None:
        file_path = os.path.join(self.output_dir, self.src_dir, f"{name}.c")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.writelines(lines)
            
    def write_include(self, name: str, lines: list[str]) -> None:
        file_path = os.path.join(self.output_dir, self.include_dir, f"{name}.h")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.writelines(lines)
        
    def generate(self, model: Model) -> None:
        output_code = OutputCode(self.output_dir)
        self.generate_kernel(model, output_code)
        self.output_kernel(output_code)
        self.output_ch_conv(output_code)
        self.output_vec_mul(output_code)
        self.output_const(output_code)
        
        raise NotImplementedError()

    def generate_kernel(self, model: Model, output_code: OutputCode) -> None:
        for op in model.operators.values():
            match op:
                case Conv2D():
                    generate_conv2d(model, op, output_code)
                case DepthConv2D():
                    generate_depthwise_conv2d(model, op, output_code)
                case Add():
                    generate_add(model, op, output_code)
                case AvgPool2D():
                    generate_avgpool(model, op, output_code)
                case Reshape():
                    pass
                case _:
                    raise NotImplementedError(op.op_type)
                
    def output_kernel(self, output_code: OutputCode) -> None:
        # generate header file
        lines = [f"#include <stdint.h>\n"]
        for _, kernel in output_code.kernels.items():
            if len(kernel.definition) == 0:
                continue
            lines.append(f"{kernel.definition};\n")
        self.write_include(self.kernel_file_name, lines)
        # generate kernel source files
        for idx, kernel in output_code.kernels.items():
            if len(kernel.definition) == 0:
                continue
            lines = [
                f"#include <stdint.h>\n"
                f"#include \"{self.vec_mul_file_name}.h\"\n",
                f"#include \"{self.ch_conv_file_name}.h\"\n",
                f"#include \"{self.kernel_file_name}.h\"\n",
            ]
            for const_name, _ in kernel.const:
                lines.append(f"extern {const_name};\n")
            lines.extend([
                f"{kernel.definition}{{\n" ,
                kernel.content,
                f"}}\n"
            ])
            self.write_src(f"{self.layer_file_name_base}{idx}", lines)
           
    def output_vec_mul(self, output_code: OutputCode) -> None:
        # generate header file
        lines = [f"#include <stdint.h>\n"]
        for vec_mul in output_code.vec_mul.values():
            lines.append(f"{vec_mul.get_def()};\n")
        self.write_include(self.vec_mul_file_name, lines)
        # generate source file
        lines = [
            f"#include <stdint.h>\n"
            f"#include \"{self.intrin_file_name}.h\"\n",
        ]        
        for vec_mul in output_code.vec_mul.values():
            lines.append(f"{vec_mul.get_def()} {{\n")
            lines.append(f"{generate_vec_mul_def(vec_mul)}\n")
            lines.append(f"}}\n")
        self.write_src(self.vec_mul_file_name, lines)
            
    def output_ch_conv(self, output_code: OutputCode) -> None:
        # generate header file
        lines = [f"#include <stdint.h>"]
        for ch_conv in output_code.ch_conv.values():
            lines.append(f"{ch_conv.get_def()};\n")
        self.write_include(self.ch_conv_file_name, lines)
        # generate source file
        lines = [
            f"#include <stdint.h>\n"
            f"#include \"{self.intrin_file_name}.h\"\n",
        ]        
        for ch_conv in output_code.ch_conv.values():
            lines.append(f"{ch_conv.get_def()} {{\n")
            lines.append(f"{generate_ch_conv_def(ch_conv)}")
            lines.append(f"}}\n")
        self.write_src(self.ch_conv_file_name, lines)
            
    def output_const(self, output_code: OutputCode) -> None:
        lines = ["#include <stdint.h>"]
        for kernel in output_code.kernels.values():
            for declare, data in kernel.const:
                elements = [str(element) for element in data.flatten()]
                data_str = ", ".join(elements)
                lines.append(f"{declare} = {{{elements}}};\n")
        self.write_include(self.const_file_name, lines)
