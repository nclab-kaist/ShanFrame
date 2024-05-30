from .output_code import ChConvFunc
from .dep_conv2d_code_pieces import *
from .utils import indent_lines


def generate_ch_conv_def(func: ChConvFunc) -> str:
    match func.kernel_size, func.stride, func.rev:
        case 5| 3, 1 | 2, _: return chconv_preset_content(func.kernel_size, func.stride, func.rev, 1)
        case _, _, _: return chconv_generic_content(func.kernel_size, func.stride, func.rev, 1)


def test():
    test_func = ChConvFunc(3, 1, True)
    print(f"{test_func.get_def()}{{")
    print(generate_ch_conv_def(test_func))
    print("}")