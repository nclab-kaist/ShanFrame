from .output_code import VecMulFunc
from .utils import indent_lines
from ..ir import DataLayout


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


def c2o2_mac_setup(col_size: int, indent: int) -> str:
    setup_str = f"""
    const int8_t *ip_b0 = input;
    const int8_t *ip_b1 = ip_b0 + {col_size};
    const int32_t contrib_0 = *(contrib_p++);
    const int32_t contrib_1 = *(contrib_p++);
    const float scale_0 = *(scales_p++);
    const float scale_1 = *(scales_p++);
    
    int32_t ch_0_out_0 = contrib_0;
    int32_t ch_0_out_1 = contrib_0;
    int32_t ch_1_out_0 = contrib_1;
    int32_t ch_1_out_1 = contrib_1;
    int32_t val0, val1, val2, val3, val4, val5;
    """
    return indent_lines(setup_str, indent)


def c2o2_mac_4(is_head: bool, is_tail: bool, indent: int) -> str:
    mac_head = """
        val1 = arm_nn_read_q7x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val4 = arm_nn_read_q7x4_ia(&ip_b1);
    """ if is_head else """
        val1 = arm_nn_read_q7x4_ia(&ip_b0);
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = arm_nn_read_q7x4_ia(&ip_b1);
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
    """
    mac_body = """
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val0 = arm_nn_read_q7x4_ia(&ip_a1);
        val3 = __SXTB16(val0);
        val0 = __SXTB16_RORn(val0, 8);
        ch_1_out_0 = __SMLAD(val3, val2, ch_1_out_0);
        ch_1_out_1 = __SMLAD(val3, val5, ch_1_out_1);
        ch_1_out_0 = __SMLAD(val0, val1, ch_1_out_0);
    """
    mac_tail = """
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
    """ if is_tail else ""

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c2o2_mac_3(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = """
        val1 = arm_nn_read_q7x4(ip_b0);
        ip_b0 += 3;
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4(ip_a0);
        ip_a0 += 3;
        val3 = __SXTB16(val0);
        val4 = arm_nn_read_q7x4(ip_b1);
        ip_b1 += 3;
    """ if is_head else """
        val1 = arm_nn_read_q7x4(ip_b0);
        ip_b0 += 3;
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = arm_nn_read_q7x4(ip_b1);
        ip_b1 += 3;
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4(ip_a0);
        ip_a0 += 3;
        val3 = __SXTB16(val0);
    """
    mac_body = """
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);
        val0 = arm_nn_read_q7x4(ip_a1);
        ip_a1 += 3;
        val3 = __SXTB16(val0);
        val0 = __SXTB16_RORn(val0, 8);
        ch_1_out_0 = __SMLAD(val3, val2, ch_1_out_0);
        ch_1_out_1 = __SMLAD(val3, val5, ch_1_out_1);
        ch_1_out_0 = __SMLABB(val0, val1, ch_1_out_0);
    """
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = """
        ch_1_out_1 = __SMLABB(val0, val4, ch_1_out_1);
    """

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c2o2_mac_2(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = """
        val1 = arm_nn_read_q7x4(ip_b0);
        ip_b0 += 2;
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);
        val4 = arm_nn_read_q7x4(ip_b1);
        ip_b1 += 2;
    """ if is_head else """
        val1 = arm_nn_read_q7x4(ip_b0);
        ip_b0 += 2;
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = arm_nn_read_q7x4(ip_b1);
        ip_b1 += 2;
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);
    """
    mac_body = """
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        val1 = __PKHBT_LSLn(val1, val2, 16); //b00 b01
        val0 = __PKHBT_LSLn(val0, val3, 16); //a00 a01
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val5 = __SXTB16(val4);
        val4 = __SXTB16_RORn(val4, 8);
        val4 = __PKHBT_LSLn(val4, val5, 16); //b10, b11
        ch_0_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val2 = arm_nn_read_q7x4(ip_a1);
        ip_a1 += 2;
        val3 = __SXTB16(val0);
        val2 = __SXTB16_RORn(val2, 8);
        val2 = __PKHBT_LSLn(val2, val3, 16); //a10, a11
        ch_1_out_0 = __SMLAD(val2, val1, ch_1_out_0);
    """
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = """
        ch_1_out_1 = __SMLAD(val2, val4, ch_1_out_1);
    """

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c2o2_mac_1(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = """
        val1 = *(ip_b0++);
        val0 = *(ip_a0++);
        val4 = *(ip_b1++);
    """ if is_head else """
        val1 = *(ip_b0++);
        ch_1_out_1 = __SMLAD(val0, val4, ch_1_out_1);
        val4 = *(ip_b1++);
        val0 = *(ip_a0++);
    """
    mac_body = """
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);
        val5 = *(ip_a1++);
        ch_1_out_0 = __SMLABB(val5, val1, ch_1_out_0);
    """
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = """
        ch_1_out_1 = __SMLABB(val5, val4, ch_1_out_1);
    """

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c2o2_output(output_layout: DataLayout, indent: int) -> str:
    output_setup = f"""
        const int8_t activation_max = 127;
        const int8_t activation_min = -128;
    """
    out_elements = [["ch_0_out_0", "ch_0_out_1"], ["ch_1_out_0", "ch_1_out_1"]]
    scales = ["scale_0", "scale_1"]
    out_locs = ["out_0", "out_1"]

    # requantization
    requant_str = ""
    for ch in range(0, 2):
        for out in range(0, 2):
            element = out_elements[ch][out]
            scale = scales[ch]
            requant_str += f"""
            
                {element} = (int32_t)((float){element} * {scale});
                {element} += out_offset;
                {element} = MAX({element}, activation_min);
                {element} = MIN({element}, activation_max);
            """

    # write output
    out_str = ""
    match output_layout:
        case DataLayout.HWC:
            out_str = """
                *out_0++ = (int8_t)ch_0_out_0;
                *out_1++ = (int8_t)ch_0_out_1;
                *out_0++ = (int8_t)ch_1_out_0;
                *out_1++ = (int8_t)ch_1_out_1;
            """
        case DataLayout.CHW:
            out_str = """
                *out_0++ = (int8_t)ch_0_out_0;
                *out_0++ = (int8_t)ch_0_out_1;
                *out_1++ = (int8_t)ch_1_out_0;
                *out_1++ = (int8_t)ch_1_out_1;
                out_0 += 2 * ch_offset - 2;
                out_1 += 2 * ch_offset - 2;
            """
        case _:
            raise NotImplementedError(
                f"unsupported data type: {output_layout}")

    return indent_lines(output_setup + requant_str + out_str, indent)


def c1o2_mac_setup(col_size: int, indent: int) -> str:
    setup_str = f"""
    const int8_t *ip_b0 = input;
    const int8_t *ip_b1 = ip_b0 + {col_size};
    const int32_t contrib_0 = *(contrib_p++);
    const float scale_0 = *(scales_p++);
    
    int32_t ch_0_out_0 = contrib_0;
    int32_t ch_0_out_1 = contrib_0;
    int32_t val0, val1, val2, val3, val4, val5;
    """
    return indent_lines(setup_str, indent)


def c1o2_mac_4(is_head: bool, is_tail: bool, indent: int) -> str:
    mac_head = """
        val1 = arm_nn_read_q7x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val4 = arm_nn_read_q7x4_ia(&ip_b1);
    """ if is_head else """
        val1 = arm_nn_read_q7x4_ia(&ip_b0);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = arm_nn_read_q7x4_ia(&ip_b1);
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
    """
    mac_body = """
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);
    """
    mac_tail = """
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
    """ if is_tail else ""

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c1o2_mac_3(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = """
        val1 = arm_nn_read_q7x4_ia(&ip_b0);
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
        val4 = arm_nn_read_q7x4_ia(&ip_b1);
    """ if is_head else """
        val1 = arm_nn_read_q7x4_ia(&ip_b0);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = arm_nn_read_q7x4_ia(&ip_b1);
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4_ia(&ip_a0);
        val3 = __SXTB16(val0);
    """
    mac_body = """
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        ch_0_out_0 = __SMLAD(val3, val2, ch_0_out_0);
        val5 = __SXTB16(val4);
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        val4 = __SXTB16_RORn(val4, 8);
        ch_0_out_1 = __SMLAD(val3, val5, ch_0_out_1);
    """
    mac_tail = """
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);
    """ if is_tail else ""

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c1o2_mac_2(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = """
        val1 = arm_nn_read_q7x4(ip_b0);
        ip_b0 += 2;
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);
        val4 = arm_nn_read_q7x4(ip_b1);
        ip_b1 += 2;
    """ if is_head else """
        val1 = arm_nn_read_q7x4(ip_b0);
        ip_b0 += 2;
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = arm_nn_read_q7x4(ip_b1);
        ip_b1 += 2;
        val2 = __SXTB16(val1);
        val0 = arm_nn_read_q7x4(ip_a0);
        ip_a0 += 2;
        val3 = __SXTB16(val0);
    """
    mac_body = """
        val1 = __SXTB16_RORn(val1, 8);
        val0 = __SXTB16_RORn(val0, 8);
        val1 = __PKHBT_LSLn(val1, val2, 16); //b00 b01
        val0 = __PKHBT_LSLn(val0, val3, 16); //a00 a01
        ch_0_out_0 = __SMLAD(val0, val1, ch_0_out_0);
        val5 = __SXTB16(val4);
        val4 = __SXTB16_RORn(val4, 8);
        val4 = __PKHBT_LSLn(val4, val5, 16); //b10, b11
    """
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = """
        ch_0_out_1 = __SMLAD(val0, val4, ch_1_out_1);
    """

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c1o2_mac_1(is_head: bool, is_tail: bool, indent: int) -> str:
    # XXX: if not is_head, the previous mac must be mac_4
    mac_head = """
        val1 = *(ip_b0++);
        val0 = *(ip_a0++);
        val4 = *(ip_b1++);
    """ if is_head else """
        val1 = *(ip_b0++);
        ch_0_out_1 = __SMLAD(val0, val4, ch_0_out_1);
        val4 = *(ip_b1++);
        val0 = *(ip_a0++);
    """
    mac_body = """
        ch_0_out_0 = __SMLABB(val0, val1, ch_0_out_0);
        
    """
    assert is_tail, "SIMD MAC non 4 must be tail"
    mac_tail = """
        ch_0_out_1 = __SMLABB(val0, val4, ch_0_out_1);
    """

    return indent_lines(mac_head + mac_body + mac_tail, indent)


def c1o2_output(output_layout: DataLayout, indent: int) -> str:
    output_setup = f"""
        const int8_t activation_max = 127;
        const int8_t activation_min = -128;
    """

    out_elements = ["ch_0_out_0", "ch_0_out_1"]

    # requantization
    requant_str = ""
    for out in range(0, 2):
        element = out_elements[out]
        scale = "scale_0"
        requant_str += f"""
        
            {element} = (int32_t)((float){element} * {scale});
            {element} += out_offset;
            {element} = MAX({element}, activation_min);
            {element} = MIN({element}, activation_max);
        """

    # write output
    out_str = ""
    match output_layout:
        case DataLayout.HWC:
            out_str = """
                *out_0++ = (int8_t)ch_0_out_0;
                *out_1++ = (int8_t)ch_0_out_1;
            """
        case DataLayout.CHW:
            out_str = """
                *out_0++ = (int8_t)ch_0_out_0;
                *out_0++ = (int8_t)ch_0_out_1;
                out_0 += 2 * ch_offset - 2;
            """
        case _:
            raise NotImplementedError(
                f"unsupported data type: {output_layout}")

    return indent_lines(output_setup + requant_str + out_str, indent)


def conv2d_1x1_setup(num_elements: int, output_layout: DataLayout, indent: int) -> str:
    ch_offset = "1" if output_layout == DataLayout.HWC else "num_elements"
    return indent_lines(f"""
        const int num_elements = {num_elements};
        const int8_t *input_elem;
        const int ch_offset = {ch_offset}
        int8_t *out = output;
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
        out = {o2_vec_mul_call};
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
        for (int i = {num}; i > 0; i -= 1) {{
    """, indent)
    indent += 1
    loop_body = indent_lines(f"""
        input_elem -= 2 * input_c;                     
        out = {o2_vec_mul_call};
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
        out = {o1_vec_mul_call};
    """, indent)
