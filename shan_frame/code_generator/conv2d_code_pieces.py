from .output_code import OutputCode, VecMulFunc
from .utils import *
from ..ir import DataLayout, Tensor


def c2o2_setup(col_size: int, output_layout: DataLayout, indent: int) -> str:
    setup_str = f"""
        int8_t *out_0 = output;
        const int32_t *contrib_p = contrib;
        const float *scales_p = scales;
        const int8_t *ip_a0 = weight;
        const int8_t *ip_a1 = ip_a0 + {col_size};
    """
    match output_layout:
        case DataLayout.HWC:
            setup_str += "int8_t *out_1 = out_0 + row_count;"
        case DataLayout.CHW:
            setup_str += "int8_t *out_1 = out_0 + ch_offset;"
    return indent_lines(setup_str, indent)


def mac_setup(c: int, o: int, col_size: int, indent: int) -> str:
    content = ""
    for j in range(0, o):
        content += indent_lines(f"const int8_t *ip_b{j} = input + {j} * {col_size};", indent)
    for i in range(0, c):
        content += indent_lines(f"""
            const int32_t contrib_{i} = *(contrib_p++);
            const float scale_{i} = *(scales_p++);""", indent)
        for j in range(0, o):
            content += indent_lines(
                f"int32_t ch_{i}_out_{j} = contrib_{i};", indent)
    content += indent_lines(f"int32_t val0, val1, val2, val3, val4, val5;", indent)
    return content
    

def c2o1_mac_4(indent: int) -> str:
    return indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val0 = read_int8x4_ia(&ip_a1);
        val3 = __SXTB16(val0);
        val0 = __SXTB16_RORn(val0, 8);
        ch_1_out_0 = __SMLAD(val3, val2, ch_1_out_0);
        ch_1_out_0 = __SMLAD(val0, val1, ch_1_out_0);""", indent)


def c2o2_mac_4(is_head: bool, is_tail: bool, indent: int) -> str:
    mac_head = indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val4 = read_int8x4_ia(&ip_b1);""" if is_head else """
        val1 = read_int8x4_ia(&ip_b0);
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = read_int8x4_ia(&ip_b1);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);""", indent)
    mac_body = indent_lines("""
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val0 = read_int8x4_ia(&ip_a1);
        val3 = __SXTB16(val0);
        val0 = __SXTB16_RORn(val0, 8);
        ch_1_out_0 = __SMLAD(val3, val2, ch_1_out_0);
        ch_1_out_1 = __SMLAD(val3, val5, ch_1_out_1);
        ch_1_out_0 = __SMLAD(val0, val1, ch_1_out_0);""", indent)
    mac_tail = indent_lines("""
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);""" if is_tail else "", indent)

    return mac_head + mac_body + mac_tail


def c2o1_mac_3(indent: int) -> str:
    return indent_lines(f"""
        val1 = read_int8x4(ip_b0);
        ip_b0 += 3;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 3;
        val3 = __SXTB16(val0);
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        val0 = read_int8x4(ip_a1);
        ip_a1 += 3;
        val3 = __SXTB16(val0);
        val0 = __SXTB16_RORn(val0, 8);
        ch_1_out_0 = __SMLAD(val3, val2, ch_1_out_0);
        ch_1_out_0 = __SMLABB(val0, val1, ch_1_out_0);""", indent)


def c2o2_mac_3(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = indent_lines("""
        val1 = read_int8x4(ip_b0);
        ip_b0 += 3;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 3;
        val3 = __SXTB16(val0);
        val4 = read_int8x4(ip_b1);
        ip_b1 += 3;""" if is_head else """
        val1 = read_int8x4(ip_b0);
        ip_b0 += 3;
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = read_int8x4(ip_b1);
        ip_b1 += 3;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 3;
        val3 = __SXTB16(val0);""", indent)
    mac_body = indent_lines("""
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);
        val0 = read_int8x4(ip_a1);
        ip_a1 += 3;
        val3 = __SXTB16(val0);
        val0 = __SXTB16_RORn(val0, 8);
        ch_1_out_0 = __SMLAD(val3, val2, ch_1_out_0);
        ch_1_out_1 = __SMLAD(val3, val5, ch_1_out_1);
        ch_1_out_0 = __SMLABB(val0, val1, ch_1_out_0);""", indent)
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = indent_lines("""
        ch_1_out_1 = __SMLABB(val0, val4, ch_1_out_1);""", indent)

    return mac_head + mac_body + mac_tail


def c2o1_mac_2(indent: int) -> str:
    return indent_lines(f"""
        val1 = read_int8x4(ip_b0);
        ip_b0 += 2;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);    
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        val1 = __PKHBT_LSLn(val1, val2, 16); //b00 b01
        val0 = __PKHBT_LSLn(val0, val3, 16); //a00 a01
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val2 = read_int8x4(ip_a1);
        ip_a1 += 2;
        val3 = __SXTB16(val0);
        val2 = __SXTB16_RORn(val2, 8);
        val2 = __PKHBT_LSLn(val2, val3, 16); //a10, a11
        ch_1_out_0 = __SMLAD(val2, val1, ch_1_out_0);""", indent)


def c2o2_mac_2(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = indent_lines("""
        val1 = read_int8x4(ip_b0);
        ip_b0 += 2;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);
        val4 = read_int8x4(ip_b1);
        ip_b1 += 2;""" if is_head else """
        val1 = read_int8x4(ip_b0);
        ip_b0 += 2;
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = read_int8x4(ip_b1);
        ip_b1 += 2;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);""", indent)
    mac_body = indent_lines("""
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        val1 = __PKHBT_LSLn(val1, val2, 16); //b00 b01
        val0 = __PKHBT_LSLn(val0, val3, 16); //a00 a01
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val5 = __SXTB16(val4);
        val4 = __SXTB16_RORn(val4, 8);
        val4 = __PKHBT_LSLn(val4, val5, 16); //b10, b11
        ch_0_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val2 = read_int8x4(ip_a1);
        ip_a1 += 2;
        val3 = __SXTB16(val0);
        val2 = __SXTB16_RORn(val2, 8);
        val2 = __PKHBT_LSLn(val2, val3, 16); //a10, a11
        ch_1_out_0 = __SMLAD(val2, val1, ch_1_out_0);""", indent)
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = indent_lines("""
        ch_1_out_1 = __SMLAD(val2, val4, ch_1_out_1);""", indent)

    return mac_head + mac_body + mac_tail


def c2o1_mac_1(indent: int) -> str:
    return indent_lines("""
        val1 = *(ip_b0++);
        val0 = *(ip_a0++);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        val5 = *(ip_a1++);
        ch_1_out_0 = __SMLABB(val5, val1, ch_1_out_0);
    """, indent)


def c2o2_mac_1(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = indent_lines("""
        val1 = *(ip_b0++);
        val0 = *(ip_a0++);
        val4 = *(ip_b1++);""" if is_head else """
        val1 = *(ip_b0++);
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = *(ip_b1++);
        val0 = *(ip_a0++);""", indent)
    mac_body = indent_lines("""
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);
        val5 = *(ip_a1++);
        ch_1_out_0 = __SMLABB(val5, val1, ch_1_out_0);""", indent)
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = indent_lines("""
        ch_1_out_1 = __SMLABB(val5, val4, ch_1_out_1);""", indent)

    return mac_head + mac_body + mac_tail
        

def mac_output(c: int, o: int, output_layout: DataLayout, indent: int) -> str:
    content = """
        const int8_t activation_max = 127;
        const int8_t activation_min = -128;"""
    # requantization
    for i in range(0, c):
        for j in range(0, o):
            element = f"ch_{i}_out_{j}"
            scale = f"scale_{i}"
            content += f"""
                {element} = (int32_t)((float){element} * {scale});
                {element} += out_offset;
                {element} = MAX({element}, activation_min);
                {element} = MIN({element}, activation_max);"""
    # write output
    for i in range(0, c):
        for j in range(0, o):
            element = f"ch_{i}_out_{j}"
            pos = f"out_{j}" if output_layout == DataLayout.HWC else f"out_{i}"
            content += f"*({pos}++) = (int8_t){element};\n"
        if output_layout == DataLayout.CHW:
            pos = f"out_{i}"
            content += f"{pos} += {c} * ch_offset - {o};\n"

    return indent_lines(content, indent)


def c1o1_mac_4(indent: int) -> str:
    return indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
    """, indent)


def c1o2_mac_4(is_head: bool, is_tail: bool, indent: int) -> str:
    mac_head = indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val4 = read_int8x4_ia(&ip_b1);""" if is_head else """
        val1 = read_int8x4_ia(&ip_b0);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = read_int8x4_ia(&ip_b1);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);""", indent)
    mac_body = indent_lines("""
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);""", indent)
    mac_tail = indent_lines("""
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);""" if is_tail else "", indent)

    return mac_head + mac_body + mac_tail


def c1o1_mac_3(indent: int) -> str:
    return indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
    """, indent)


def c1o2_mac_3(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val4 = read_int8x4_ia(&ip_b1);""" if is_head else """
        val1 = read_int8x4_ia(&ip_b0);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = read_int8x4_ia(&ip_b1);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);""", indent)
    mac_body = indent_lines("""
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);""", indent)
    mac_tail = indent_lines("""
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);""" if is_tail else "", indent)

    return mac_head + mac_body + mac_tail


def c1o1_mac_2(indent: int) -> str:
    return indent_lines("""
        val1 = read_int8x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = read_int8x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        val1 = __PKHBT_LSLn(val1, val2, 16); //b00 b01
        val0 = __PKHBT_LSLn(val0, val3, 16); //a00 a01
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);""", indent)


def c1o2_mac_2(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = indent_lines("""
        val1 = read_int8x4(ip_b0);
        ip_b0 += 2;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);
        val4 = read_int8x4(ip_b1);
        ip_b1 += 2;""" if is_head else """
        val1 = read_int8x4(ip_b0);
        ip_b0 += 2;
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = read_int8x4(ip_b1);
        ip_b1 += 2;
        val2 = __SXTB16(val1);
        val0 = read_int8x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);""", indent)
    mac_body = indent_lines("""
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        val1 = __PKHBT_LSLn(val1, val2, 16); //b00 b01
        val0 = __PKHBT_LSLn(val0, val3, 16); //a00 a01
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val5 = __SXTB16(val4);
        val4 = __SXTB16_RORn(val4, 8);
        val4 = __PKHBT_LSLn(val4, val5, 16); //b10, b11""", indent)
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = indent_lines("""
        ch_0_out_1 = __SMLAD(val0, val4, ch_1_out_1);""", indent)

    return mac_head + mac_body + mac_tail


def c1o1_mac_1(indent: int) -> str:
    return indent_lines("""
        val1 = *(ip_b0++);
        val0 = *(ip_a0++);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);""", indent)


def c1o2_mac_1(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = indent_lines("""
        val1 = *(ip_b0++);
        val0 = *(ip_a0++);
        val4 = *(ip_b1++);""" if is_head else """
        val1 = *(ip_b0++);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = *(ip_b1++);
        val0 = *(ip_a0++);""", indent)
    mac_body = indent_lines("""
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);""", indent)
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = indent_lines("""
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);""", indent)

    return mac_head + mac_body + mac_tail


def mac_subloop_16(c: int, o: int, mac: int, indent: int) -> str:
    assert mac % 16 == 0
    loop_count = mac // 16
    content = indent_lines(f"for(int mac16_count = {loop_count}; mac16_count > 0; mac16_count--){{", indent)
    indent += 1
    content += mac_body(c, o, 16, indent)
    indent -= 1
    content += indent_lines("}", indent)
    return content


def mac_body(c: int, o: int, mac: int, indent: int, head: bool = True) -> str:
    if mac >= 48:
        current_mac = (mac // 16) * 16
        remain_mac = mac - current_mac
        this_round = mac_subloop_16(c, o, current_mac, indent)
        return this_round + mac_body(c, o, remain_mac, indent)
    if mac == 0:
        return ""
    this_round = ""    
    current_mac = min(mac, 4)
    remain_mac = mac - current_mac   
    match c, o, current_mac:
        case 2, 2, 4: this_round = c2o2_mac_4(head, remain_mac==0, indent)
        case 2, 2, 3: this_round = c2o2_mac_3(head, remain_mac==0, indent)
        case 2, 2, 2: this_round = c2o2_mac_2(head, remain_mac==0, indent)
        case 2, 2, 1: this_round = c2o2_mac_1(head, remain_mac==0, indent)
        case 1, 2, 4: this_round = c1o2_mac_4(head, remain_mac==0, indent)
        case 1, 2, 3: this_round = c1o2_mac_3(head, remain_mac==0, indent)
        case 1, 2, 2: this_round = c1o2_mac_2(head, remain_mac==0, indent)
        case 1, 2, 1: this_round = c1o2_mac_1(head, remain_mac==0, indent)
        case 2, 1, 4: this_round = c2o1_mac_4(indent)
        case 2, 1, 3: this_round = c2o1_mac_3(indent)
        case 2, 1, 2: this_round = c2o1_mac_2(indent)
        case 2, 1, 1: this_round = c2o1_mac_1(indent)
        case 1, 1, 4: this_round = c1o1_mac_4(indent)
        case 1, 1, 3: this_round = c1o1_mac_3(indent)
        case 1, 1, 2: this_round = c1o1_mac_2(indent)
        case 1, 1, 1: this_round = c1o1_mac_1(indent)
        case _, _, _: raise NotImplementedError(f"{c}, {o}, {current_mac}")
    next_round = "" if remain_mac == 0 else mac_body(c, o, remain_mac, indent, False)
    return this_round + next_round


def conv2d_setup(
    idx: int,
    input: Tensor,
    output: Tensor,
    indent: int
) -> str:
    match output.layout:
        case DataLayout.HWC:
            ch_offset = 1
            out_update = output.dim_c
        case DataLayout.CHW:
            ch_offset = output.mem_size() // output.dim_c
            out_update = 1
        case _: raise RuntimeError(f"unsupported data layout: {output.layout}")
    return indent_lines(f"""
        const int out_c = {output.dim_c};
        const int ch_offset = {ch_offset};
        const int out_update = {out_update};
        const int8_t input_zero_point = {input.zero_point[0]};
        const int8_t out_offset = {output.zero_point[0]};
        const int8_t *weight = {weight_name(idx)};
        const float *scales = {scales_name(idx)};
        const int32_t *contrib = {contrib_name(idx)};
        int8_t *pad_pos = output;
        const int8_t *input_elem;
        int8_t *out;
    """, indent)


def pad_output(offset: int, size: int, indent) -> str:
    if size == 0:
        return ""
    return indent_lines(f"memset(&output[{offset}], out_offset, {size});\n", indent)


def conv2d_1x1_setup(out_c: int, output_layout: DataLayout, indent: int) -> str:
    ch_offset = "1" if output_layout == DataLayout.HWC else "num_elements"
    out_update = out_c if output_layout == DataLayout.HWC else 1
    return indent_lines(f"""
        const int ch_offset = {ch_offset}
        const int out_update = {out_update};
        const int out_offset = 
        const int8_t *input_elem;
        int8_t *out;
    """, indent)


def conv2d_1x1_even_loop_low_to_high(
    start: int, num: int, out_c: int, o2_vec_mul: VecMulFunc, weight: str, scales: str, contrib: str, out_offset: str, indent: int
) -> str:

    o2_vec_mul_call = o2_vec_mul.get_call(
        "input_elem", "out", weight,
        str(out_c), "ch_offset",
        out_offset, scales, contrib
    )
    setup = indent_lines(f"""
        input_elem = input + {start};
        for (int i = {num}; i > 0; i -= 1) {{
    """, indent)
    indent += 1
    loop_body = indent_lines(f"""
        {o2_vec_mul_call};
        out += 2 * out_update;
        input_elem += 2 * input_c;
    """, indent)
    indent -= 1
    cleanup = indent_lines("}\n", indent)
    return setup + loop_body + cleanup


def conv2d_1x1_even_loop_high_to_low(
    start: int, num: int, out_c: int, o2_vec_mul: VecMulFunc, weight: str, scales: str, contrib: str, out_offset: str, indent: int
) -> str:
    o2_vec_mul_call = o2_vec_mul.get_call(
        "input_elem", "out", weight,
        str(out_c), "ch_offset",
        out_offset, scales, contrib
    )
    setup = indent_lines(f"""
        input_elem = input + {start};
        out = output + {start} / input_c * {out_c};
        for (int i = {num}; i > 0; i -= 1) {{
    """, indent)
    indent += 1
    loop_body = indent_lines(f"""
        input_elem -= 2 * input_c;
        out -= 2 * out_update;
        {o2_vec_mul_call};
    """, indent)
    indent -= 1
    cleanup = indent_lines("}\n", indent)
    return setup + loop_body + cleanup


def conv2d_1x1_odd_cleanup(out_c: int, o1_vec_mul: VecMulFunc, weight: str, scales: str, contrib: str, out_offset: str, indent: int) -> str:
    o1_vec_mul_call = o1_vec_mul.get_call(
        "input_elem", "out", weight,
        str(out_c), "ch_offset",
        out_offset, scales, contrib
    )
    return indent_lines(f"""
        {o1_vec_mul_call};
    """, indent)


def conv2d_prepad(output: Tensor, indent: int) -> str:
    content = ""
    # set prepad
    if output.layout == DataLayout.HWC:
        pad_row_size = output.prepad_h * \
            (output.dim_w + 2 * output.prepad_w) * output.dim_c
        if output.prepad_h != 0:
            # prepad row
            content += indent_lines(
                f"memset(&output[0], out_offset, {pad_row_size});\n", indent)
            content += indent_lines(
                f"memset(&output[{output.mem_size() - pad_row_size}], out_offset, {pad_row_size});\n", indent)
        if output.prepad_w != 0:
            # prepad column
            content += indent_lines(
                f"pad_pos = &output[{pad_row_size}];\n", indent)
            content += indent_lines(f"""for(int i = 0; i < {output.dim_h}; i++){{
                memset(pad_pos, out_offset, {output.prepad_w * output.dim_c});
                memset(pad_pos + {(output.prepad_w + output.dim_w) * output.dim_c}, out_offset, {output.prepad_w * output.dim_c});
                pad_pos += {(output.dim_w + 2 * output.prepad_w) * output.dim_c};
            }}""", indent)
    elif output.layout == DataLayout.CHW:
        pad_row_size = (output.prepad_h * (output.dim_w + 2 * output.prepad_w))
        if output.prepad_h != 0:
            pad_shift = output.dim_h * \
                (output.dim_w + 2 * output.prepad_w) + 2 * output.prepad_w
            content += indent_lines(f"""
            pad_pos = output;
            for(int i = 0; i < {output.dim_c}; i++) {{
                memset(pad_pos, out_offset, {pad_row_size});
                memset(pad_pos + {(output.dim_h + output.prepad_h) * (output.dim_w + 2 * output.prepad_w)}, out_offset, {pad_row_size});
                pad_pos += {(output.dim_w + 2 * output.prepad_w) * (output.dim_h + 2 * output.prepad_h)};
            }}
            """, indent)
        if output.prepad_w != 0:
            content += indent_lines(f"""
                pad_pos = output + {pad_row_size};
                for (int i = 0; i < {output.dim_c}; i++){{
                    for (int j = 0; j < {output.dim_h}; j++) {{
                        memset(pad_pos, out_offset, {output.prepad_w});
                        memset(pad_pos + {output.dim_w + output.prepad_w}, out_offset, {output.prepad_w});
                        pad_pos += {output.dim_w + 2 * output.prepad_w};
                    }}
                    pad_pos += {(output.dim_w + 2 * output.prepad_w) * 2 * output.prepad_h};
                }}
            """, indent)
    return content


def conv2d_1x1_by_row(input: Tensor, output: Tensor, output_code: OutputCode, indent) -> str:
    content = ""
    out_start = output.prepad_h * \
        (output.dim_w + 2 * output.prepad_w) + output.prepad_w
    if output.layout == DataLayout.HWC:
        out_start *= output.dim_c
    o2_vec_mul = output_code.add_vec_mul(2, input.dim_c, output.layout)
    o2_call = o2_vec_mul.get_call(
        "input_elem", "out", "weight",
        str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
    )
    o1_cleanup = ""
    if output.dim_w % 2 != 0:
        o1_vec_mul = output_code.add_vec_mul(1, input.dim_c, output.layout)
        o1_call = o1_vec_mul.get_call(
            "input_elem", "out", "weight",
            str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
        )
        o1_cleanup = indent_lines(f"""
            {o1_call};
            out += out_update;
            input_elem += {input.dim_c};
        """, indent)
    content += indent_lines(f"""
        out = output + {out_start};
        input_elem = input;
        for(int out_h = 0; out_h < {output.dim_h}; out_h++){{
            for(int out_w = 0; out_w < {output.dim_w} / 2; out_w++){{
                {o2_call};
                out += 2 * out_update;
                input_elem += 2 * {input.dim_c};
            }}
            {o1_cleanup}
            out += 2 * {output.prepad_w} * out_update;
        }}
    """, indent)
    return content


def conv2d_1x1_low_to_high(input_start: int, output_start: int, num: int,
                           input: Tensor, output: Tensor, output_code: OutputCode, indent) -> str:
    content = ""
    if num == 0:
        return content
    pad_head = output.prepad_h * output.dim_w
    if output.layout == DataLayout.HWC:
        pad_head *= output.dim_c
    o2_vec_mul = output_code.add_vec_mul(2, input.dim_c, output.layout)
    o2_call = o2_vec_mul.get_call(
        "input_elem", "out", "weight",
        str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
    )
    o1_cleanup = ""
    if num % 2 != 0:
        o1_vec_mul = output_code.add_vec_mul(1, input.dim_c, output.layout)
        o1_call = o1_vec_mul.get_call(
            "input_elem", "out", "weight",
            str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
        )
        o1_cleanup = indent_lines(f"""
            {o1_call};
            out += out_update;
            input_elem += {input.dim_c};
        """, indent)
    content += indent_lines(f"""
        out = output + {pad_head + output_start};
        input_elem = input + {input_start};
        for(int i = 0; i < {num} / 2; i++){{
            {o2_call};
            out += 2 * out_update;
            input_elem += 2 * {input.dim_c};            
        }}
        {o1_cleanup}
    """, indent)
    return content


def conv2d_1x1_high_to_low(input_start: int, output_start: int, num: int,
                           input: Tensor, output: Tensor, output_code: OutputCode, indent) -> str:
    content = ""
    if num == 0:
        return content
    pad_head = output.prepad_h * output.dim_w
    if output.layout == DataLayout.HWC:
        pad_head *= output.dim_c
    o2_vec_mul = output_code.add_vec_mul(2, input.dim_c, output.layout)
    o2_call = o2_vec_mul.get_call(
        "input_elem", "out", "weight",
        str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
    )
    o1_cleanup = ""
    if num % 2 != 0:
        o1_vec_mul = output_code.add_vec_mul(1, input.dim_c, output.layout)
        o1_call = o1_vec_mul.get_call(
            "input_elem", "out", "weight",
            str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
        )
        o1_cleanup = indent_lines(f"""
            out -= out_update;
            input_elem -= {input.dim_c};            
            {o1_call};
        """, indent)
    content += indent_lines(f"""
        out = output + {pad_head + output_start};
        input_elem = input + {input_start};
        for(int i = 0; i < {num} / 2; i++){{
            out -= 2 * out_update;
            input_elem -= 2 * {input.dim_c}; 
            {o2_call}; 
        }}
        {o1_cleanup}
    """, indent)
    return content


def conv2d_1x1_all(input: Tensor, output: Tensor, output_code: OutputCode, indent) -> str:
    assert output.prepad_w == 0
    return conv2d_1x1_low_to_high(0, 0, output.dim_h * output.dim_w, input, output, output_code, indent)


def conv2d_window_slide(op: Conv2D, input: Tensor, weight: Tensor, output: Tensor, output_code: OutputCode, indent) -> str:
    content = ""
    o2_vec_mul = output_code.add_vec_mul(
        2, weight.dim_h * weight.dim_w * input.dim_c, output.layout)
    o2_call = o2_vec_mul.get_call(
        "buffer", "out", "weight",
        str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
    )
    o1_block = ""
    if output.dim_w % 2 != 0:
        o1_vec_mul = output_code.add_vec_mul(
            2, weight.dim_h * weight.dim_w * input.dim_c, output.layout)
        o1_call = o1_vec_mul.get_call(
            "buffer", "out", "weight",
            str(output.dim_c), "ch_offset", "out_offset", "scales", "contrib"
        )
        o1_block = indent_lines(f"""
            {o1_call};
            col_buf = buffer;
            out += out_update;
        """, indent)
    output_start = output.prepad_h * \
        (2 * output.prepad_w + output.dim_w) + output.prepad_w
    if output.layout == DataLayout.HWC:
        output_start *= output.dim_c
    content += indent_lines(f"""
        out = output + {output_start};
        const int in_window_x_start = -{op.pad_w};
        int in_window_y = -{op.pad_h};
        int8_t *col_buf = buffer;
        int8_t *window_row_buf = col_buf;
        for(int oy = 0; oy < {output.dim_h}; oy++){{
            int in_window_x = in_window_x_start;
            for(int ox = 0; ox < {output.dim_w}; ox++){{
    """, indent)
    indent += 2
    content += indent_lines(f"""
                int8_t *window_row_buf = col_buf;
                int cp_y = 0;
                if(in_window_y < 0) {{
                    int pad_row = MIN(-in_window_y, {weight.dim_h});
                    memset(window_row_buf, input_zero_point, {weight.dim_w * weight.dim_c} * pad_row);
                    window_row_buf += {weight.dim_w * weight.dim_c} * pad_row;
                    cp_y += pad_row;
                }}
                int pad_head = MIN(MAX(-in_window_x, 0), {weight.dim_w});
                int pad_tail = MAX(in_window_x + {weight.dim_w} - {input.dim_w}, 0);
                int copy_size = MAX({weight.dim_w} - pad_head - pad_tail, 0);
                for(; cp_y < {weight.dim_h}; cp_y++){{
                    memset(window_row_buf, input_zero_point, pad_head * {input.dim_c});
                    window_row_buf += pad_head * {input.dim_c};
                    const int8_t *in_cp = input + (in_window_y + cp_y) * {input.dim_c * input.dim_w} + (in_window_x + pad_head) * {input.dim_c};
                    memcpy(window_row_buf, in_cp, copy_size * {input.dim_c});
                    window_row_buf += copy_size * {input.dim_c};
                    memset(window_row_buf, input_zero_point, pad_tail * {input.dim_c});
                    window_row_buf += pad_tail * {input.dim_c};
                }}
                col_buf += {weight.dim_h * weight.dim_w * weight.dim_c};
                in_window_x += {op.stride_w};

                if(col_buf == buffer + {2 * weight.dim_h * weight.dim_w * input.dim_c}){{
                    {o2_call};
                    col_buf = buffer;
                    out += 2 * out_update;
                }}
    """, indent)
    indent -= 1
    out_row_update = 2 * output.prepad_w
    if output.layout == DataLayout.HWC:
        out_row_update *= output.dim_c
    content += indent_lines(f"""
            }}
            {o1_block}
            out += {out_row_update};
            in_window_y += {op.stride_h};
    """, indent)
    indent -= 1
    content += indent_lines(f"""
        }}
    """, indent)
    return content
