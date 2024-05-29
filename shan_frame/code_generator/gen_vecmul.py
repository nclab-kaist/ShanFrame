from .conv2d_code_pieces import *
from .output_code import VecMulFunc
from .utils import indent_lines


def generate_o2_vec_mul_def(func: VecMulFunc) -> str:
    # setup needed pointers
    indent = 1
    setup = c2o2_setup(func.col_size, func.output_layout, indent)
    # build c2o2 loop
    c2o2_loop_start = indent_lines("""
        for (int c2o2_loop_count = row_count / 2; c2o2_loop_count > 0; c2o2_loop_count--){
    """, indent)
    indent += 1
    c2o2_loop_body = c2o2_mac_setup(func.col_size, indent)
    mac_4_num = func.col_size // 4
    left_over = func.col_size % 4
    match left_over:
        case 3: mac_tail = c2o2_mac_3(mac_4_num == 0, True, indent)
        case 2: mac_tail = c2o2_mac_2(mac_4_num == 0, True, indent)
        case 1: mac_tail = c2o2_mac_1(mac_4_num == 0, True, indent)
        case 0:
            mac_4_num -= 1
            mac_tail = c2o2_mac_4(mac_4_num == 0, True, indent)
        case _: raise RuntimeError()
    for idx in range(0, mac_4_num):
        c2o2_loop_body += c2o2_mac_4(idx == 0, False, indent)
    c2o2_loop_body += mac_tail
    c2o2_loop_body += indent_lines(f"""
        ip_a0 += {func.col_size};
        ip_a1 += {func.col_size};
    """, indent)
    c2o2_loop_body += c2o2_output(func.output_layout, indent)
    indent -= 1
    c2o2_loop_end = indent_lines("}", indent)
    # add c1o2 block if row count is odd
    c1o2_block_start = indent_lines("""
        if (row_count %2) {
    """, indent)
    indent += 1
    c1o2_block_body = c1o2_mac_setup(func.col_size, indent)
    mac_4_num = func.col_size // 4
    left_over = func.col_size % 4
    match left_over:
        case 3: mac_tail = c1o2_mac_3(mac_4_num == 0, True, indent)
        case 2: mac_tail = c1o2_mac_2(mac_4_num == 0, True, indent)
        case 1: mac_tail = c1o2_mac_1(mac_4_num == 0, True, indent)
        case 0:
            mac_4_num -= 1
            mac_tail = c1o2_mac_4(mac_4_num == 0, True, indent)
        case _: raise RuntimeError()
    for idx in range(0, mac_4_num):
        c1o2_block_body += c1o2_mac_4(idx == 0, False, indent)
    c1o2_block_body += mac_tail
    c1o2_block_body += c1o2_output(func.output_layout, indent)
    indent -= 1
    c1o2_block_end = indent_lines("}", indent)
    return setup + c2o2_loop_start + c2o2_loop_body + c2o2_loop_end + c1o2_block_start + c1o2_block_body + c1o2_block_end


def generate_o1_vec_mul_def(func: VecMulFunc) -> str:
    raise NotImplementedError()


def generate_vec_mul_def(func: VecMulFunc) -> str:
    match func.col_num:
        case 2:
            return generate_o2_vec_mul_def(func)
        case 1:
            return generate_o1_vec_mul_def(func)
        case _:
            raise NotImplementedError(
                f"Not supported column num for vec mul: {func.col_num}")


def test():
    test_vec_mul = VecMulFunc(2, 27, DataLayout.CHW)
    print(f"{test_vec_mul.get_def()}{{")
    print(generate_vec_mul_def(test_vec_mul))
    print("}")
