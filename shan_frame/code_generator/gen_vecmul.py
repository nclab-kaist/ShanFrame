from .conv2d_code_pieces import *
from .output_code import VecMulFunc
from .utils import indent_lines


def generate_vec_mul_content(func: VecMulFunc) -> str:
    # setup needed pointers
    indent = 1
    setup = c2o2_setup(func.col_size, func.output_layout, indent)
    # build c2o2 loop
    loop_start = indent_lines("""
        for (int c2o2_loop_count = row_count / 2; c2o2_loop_count > 0; c2o2_loop_count--){""", indent)
    indent += 1
    loop_body = mac_setup(2, func.col_num, func.col_size, indent)
    loop_body += mac_body(2, func.col_num, func.col_size, indent)
    loop_body += indent_lines(f"""
        ip_a0 += {func.col_size};
        ip_a1 += {func.col_size};
    """, indent)
    loop_body += mac_output(2, func.col_num, func.output_layout, indent)
    indent -= 1
    loop_end = indent_lines("}", indent)
    # add c1o2 block if row count is odd
    c1_block_start = indent_lines("""
        if (row_count %2) {
    """, indent)
    indent += 1
    c1_block_body = mac_setup(1, func.col_num, func.col_size, indent)
    c1_block_body += mac_body(1, func.col_num, func.col_size, indent)
    c1_block_body += mac_output(1, func.col_num, func.output_layout, indent)
    indent -= 1
    c1_block_end = indent_lines("}", indent)
    return setup + loop_start + loop_body + loop_end + c1_block_start + c1_block_body + c1_block_end


def generate_vec_mul_def(func: VecMulFunc) -> str:
    match func.col_num:
        case 2 | 1:
            return generate_vec_mul_content(func)
        case _:
            raise NotImplementedError(
                f"Not supported column num for vec mul: {func.col_num}")


def test():
    test_vec_mul = VecMulFunc(2, 27, DataLayout.CHW)
    print(f"{test_vec_mul.get_def()}{{")
    print(generate_vec_mul_def(test_vec_mul))
    print("}")
