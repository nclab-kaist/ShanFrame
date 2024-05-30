from .utils import buffer_name, indent_lines
from .gen_minorop import generate_minor_declare, generate_minor_def
from .gen_intrinsics import generate_intrin
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
    inference_file_name = "inference"
    inference_input_var = "input"
    
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
            f.write(f"#ifndef {name.upper()}_H\n")
            f.write(f"#define {name.upper()}_H\n")
            f.writelines(lines)
            f.write(f"#endif\n")
        
    def generate(self, model: Model, peak_mem: int) -> None:
        output_code = OutputCode(self.output_dir, peak_mem)
        self.generate_kernel(model, output_code)
        self.output_inference(model, output_code)
        self.output_kernel(output_code)
        self.output_ch_conv(output_code)
        self.output_vec_mul(output_code)
        self.output_const(output_code)
        self.output_intrin()
        self.output_minor_op()
        
        raise NotImplementedError()

    def generate_kernel(self, model: Model, output_code: OutputCode) -> None:
        for op in model.operators.values():
            match op:
                case Conv2D():
                    generate_conv2d(self.inference_input_var, model, op, output_code)
                case DepthConv2D():
                    generate_depthwise_conv2d(self.inference_input_var, model, op, output_code)
                case Add():
                    generate_add(model, op, output_code)
                case AvgPool2D():
                    generate_avgpool(self.inference_input_var, model, op, output_code)
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
                f"#include \"{self.intrin_file_name}.h\"\n",
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
        lines = [f"#include <stdint.h>\n"]
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
        lines = ["#include <stdint.h>\n"]
        for kernel in output_code.kernels.values():
            for declare, data in kernel.const:
                elements = [str(element) for element in data.flatten()]
                data_str = ", ".join(elements)
                lines.append(f"{declare} = {{{data_str}}};\n")
        self.write_include(self.const_file_name, lines)
        
    def output_inference(self, model: Model, output_code: OutputCode) -> None:
        # generate include file
        lines = [f"#include <stdint.h>\n"]
        inference_declare = f"int8_t *{self.inference_file_name}(int8_t *{self.inference_input_var})"
        lines.append(f"{inference_declare};\n")
        self.write_include(self.inference_file_name, lines)
        # generate source file
        lines = [
            f"#include <stdint.h>\n",
            f"#include \"{self.inference_file_name}.h\"\n",
            f"#include \"{self.minor_op_file_name}.h\"\n",
            f"#include \"{self.const_file_name}.h\"\n",
            f"#include \"{self.kernel_file_name}.h\"\n",
            f"int8_t {buffer_name()}[{output_code.mem_size}];\n"
        ]
        last_op = next(reversed(model.operators.values()))
        output_addr = model.tensors[last_op.output_idx].addr
        output = f"&{buffer_name()}[{output_addr}]"
        content = f"{inference_declare}{{\n"
        indent = 1
        for idx, kernel in output_code.kernels.items():
            content += indent_lines(f"{kernel.call};", indent)
        content += indent_lines(f"return {output};", indent)
        content += "}\n"
        lines.append(content)
        self.write_src(self.inference_file_name, lines)
        

    def output_intrin(self) -> None:
        self.write_include(self.intrin_file_name, [generate_intrin()])
        
    def output_minor_op(self) -> None:
        self.write_include(self.minor_op_file_name, [generate_minor_declare()])
        lines = [f"#include \"{self.intrin_file_name}.h\"\n"]
        lines.append(generate_minor_def())
        self.write_src(self.minor_op_file_name, lines)
        