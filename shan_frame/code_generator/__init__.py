from .output_code import OutputCode
from .gen_conv2d import generate_conv2d
from .gen_dep_conv2d import generate_depthwise_conv2d
from ..ir.operator import Conv2D, DepthConv2D
from ..ir import Model
from .gen_ch_conv import test

class CodeGenerator:
    output_dir: str
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def generate(self, model: Model) -> None:
        output_code = OutputCode(self.output_dir)
        for op in model.operators.values():
            match op:
                case Conv2D():
                    generate_conv2d(model, op, output_code)
                case DepthConv2D():
                    generate_depthwise_conv2d(model, op, output_code)
                case _:
                    raise NotImplementedError(op.op_type)
        raise NotImplementedError("CodeGenerator.generate()")