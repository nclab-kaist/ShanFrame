from .output_code import ChConvFunc
from .dep_conv2d_code_pieces import *
from .utils import indent_lines


def generate_ch_conv_def(func: ChConvFunc) -> str:
    match func.kernel_size, func.stride:
        case 3, 1: return chconv_k3x3_stride1_content(1)
        case _, _: return chconv_generic_content(func.kernel_size, func.stride, 1)


def test():
    test_func = ChConvFunc(3, 1)
    print(f"{test_func.get_def()}{{")
    print(generate_ch_conv_def(test_func))
    print("}")