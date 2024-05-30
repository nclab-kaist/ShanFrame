from .conv2d_code_pieces import *
from .output_code import VecMulFunc
from .utils import indent_lines


def chconv_setup(stride: int, rev: bool, indent: int) -> str:
    if rev:
        return indent_lines(f"""
            const int8_t *input_col = input + (out_h - 1) * {stride} * input_w + out_w * {stride};
            int8_t *out = output + (out_h * out_w) * ch_offset;
        """, indent)        
    else:
        return indent_lines(f"""
            const int row_offset = input_w - out_w * {stride};
            const int8_t *input_col = input;
            int8_t *out = output;
        """, indent)
        
def chconv_mac_setup(stride: int, o: int, rev: bool, indent: int) -> str:
    content = ""
    if rev:
        content += indent_lines(f"""input_col -= {o * stride};
        const int8_t *cols_8b = input_col;""", indent)
    else:
        content += indent_lines(f"""const int8_t *cols_8b = input_col;
        input_col += {o * stride};""", indent)
    for i in range(0, o):
        content += indent_lines(f"int32_t sum{i} = contrib;", indent)
    return content

def chconv_k3x3_mac_setup(indent: int) -> str:
    return indent_lines(f"""
        int32_t k3210, k20, k31, c3210, c20, c31, c4;
        k3210 = read_int8x4(ksrc);
        k20 = __SXTB16(k3210);
        k31 = __SXTB16_RORn(k3210, 8);
        int32_t k7654, k64, k75, ka8;
        k7654 = read_int8x4(ksrc + 4);
        k64 = __SXTB16(k7654);
        k75 = __SXTB16_RORn(k7654, 8);
        ka8 = ksrc[8];""", indent)

def chconv_mac_output(o: int, rev: bool, indent: int) -> str:
    content = ""
    if rev:
        for i in reversed(range(0, o)):
            sum = f"sum{i}"
            content += indent_lines(f"""
            {sum} = (float){sum} * scale;
            {sum} += out_offset;
            {sum} = MAX({sum}, -128);
            {sum} = MIN({sum}, 127);
            out -= ch_offset;
            *out = {sum};""", indent)
    else:
        for i in (range(0, o)):
            sum = f"sum{i}"
            content += indent_lines(f"""
            {sum} = (float){sum} * scale;
            {sum} += out_offset;
            {sum} = MAX({sum}, -128);
            {sum} = MIN({sum}, 127);
            *out = {sum};
            out += ch_offset;""", indent)
    return content

def chconv_k3x3_stride1_o2_mac(rev: bool, indent: int) -> str:
    content = chconv_mac_setup(1, 2, rev, indent)
    content += indent_lines(f"""
        c3210 = read_int8x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        sum0 = __SMLAD(c20, k20, sum0); // 0*0 + 2*2
        sum0 = __SMLABB(c31, k31, sum0); // += 1*1
        sum1 = __SMLAD(c31, k20, sum1); // 1*0 + 3*2
        sum1 = __SMLATB(c20, k31, sum1); // += 2*1
        cols_8b += input_w;
        
        c3210 = read_int8x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        sum0 = __SMLABT(c20, k31, sum0); // += 0 * 3
        sum0 = __SMLABB(c31, k64, sum0); // += 1 * 4
        sum0 = __SMLATB(c20, k75, sum0); // += 2 * 5
        sum1 = __SMLABT(c31, k31, sum1); // += 1 * 3
        sum1 = __SMLATB(c20, k64, sum1); // += 2 * 4
        sum1 = __SMLATB(c31, k75, sum1); // += 3 * 5
        cols_8b += input_w;

        c3210 = read_int8x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        sum0 = __SMLABT(c20, k64, sum0); // += 0 * 6
        sum0 = __SMLABT(c31, k75, sum0); // += 1 * 7
        sum0 = __SMLATB(c20, ka8, sum0); // += 2 * 8
        sum1 = __SMLABT(c31, k64, sum1); // += 1 * 6
        sum1 = __SMLATT(c20, k75, sum1); // += 2 * 7
        sum1 = __SMLATB(c31, ka8, sum1); // += 3 * 8""", indent)
    content += chconv_mac_output(2, rev, indent)
    return content


def chconv_k3x3_stride2_o2_mac(rev: bool, indent: int) -> str:
    content = chconv_mac_setup(1, 2, rev, indent)
    content += indent_lines(f"""         
        c3210 = read_int8x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        c4 = cols_8b[4];
        sum0 = __SMLAD(c20, k20, sum0); // 0*0 + 2*2
        sum0 = __SMLABB(c31, k31, sum0); // += 1*1
        sum1 = __SMLATB(c20, k20, sum1);
        sum1 = __SMLATB(c31, k31, sum1);
        sum1 = __SMLABT(c4, k20, sum1);
        cols_8b += input_w;
        
        c3210 = read_int8x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        c4 = cols_8b[4];
        sum0 = __SMLABT(c20, k31, sum0); // += 0 * 3
        sum0 = __SMLATB(c20, k75, sum0); // += 2 * 5
        sum0 = __SMLABB(c31, k64, sum0); // += 1 * 4
        sum1 = __SMLATT(c20, k31, sum1);
        sum1 = __SMLATB(c31, k64, sum1);
        sum1 = __SMLABB(c4, k75, sum1);
        cols_8b += input_w;
        
        c3210 = read_int8x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        c4 = cols_8b[4];
        sum0 = __SMLABT(c20, k64, sum0); // += 0 * 6
        sum0 = __SMLATB(c20, ka8, sum0); // += 2 * 8
        sum0 = __SMLABT(c31, k75, sum0); // += 1 * 7
        sum1 = __SMLATT(c20, k64, sum1);
        sum1 = __SMLATT(c31, k75, sum1);
        sum1 = __SMLABB(c4, ka8, sum1);""", indent)
    content += chconv_mac_output(2, rev, indent)
    return content


def chconv_k5x5_mac_setup(indent: int) -> str:
    return indent_lines(f"""
        int32_t k3210, k20, k31, k4;
        int32_t c3210, c20, c31, c7654, c75, c64;
    """, indent)


def chconv_k5x5_stride1_o2_mac(indent: int) -> str:
    content = chconv_mac_setup(1, 2, False, indent)
    content += indent_lines("const int8_t *k = ksrc;", indent)
    for i in range(0, 5):
        content += indent_lines(f"""
            // row {i}
            c3210 = read_int8x4(cols_8b);
            c20 = __SXTB16(c3210);
            c31 = __SXTB16_RORn(c3210, 8);
            memcpy(cols_8b+4, &c64, 2); // c54 actually
            k3210 = read_int8x4(k);
            k20 = __SXTB16(k3210);
            k31 = __SXTB16_RORn(k3210, 8);
            k4 = k[4];
            sum0 = __SMLAD(c20, k20, sum0); // 00 22
            sum0 = __SMLAD(c31, k31, sum0); // 11 33
            sum0 = __SMLABB(c64, k4, sum0);   // 44
            sum1 = __SMLAD(c31, k20, sum1); // 10 + 32
            sum1 = __SMLATB(c20, k31, sum1); // 21
            sum1 = __SMLABT(c64, k31, sum1); // 43
            sum1 = __SMLATB(c64, k4, sum1); // 54
            cols_8b += input_w;
            k += 5;
        """, indent)
    content += chconv_mac_output(2, False, indent)
    return content


def chconv_k5x5_stride2_o2_mac(indent: int) -> str:
    content = chconv_mac_setup(1, 2, False, indent)
    content += indent_lines("const int8_t *k = ksrc;", indent)
    for i in range(0, 5):
        content += indent_lines(f"""
            // row {i}
            c3210 = read_int8x4(cols_8b);
            c20 = __SXTB16(c3210);
            c31 = __SXTB16_RORn(c3210, 8);
            c7654 = read_int8x4(cols_8b);
            c64 = __SXTB16(c7654);
            c75 = __SXTB16_RORn(7654, 8);
            k3210 = read_int8x4(k);
            k20 = __SXTB16(k3210);
            k31 = __SXTB16_RORn(k3210, 8);
            k4 = k[4];
            sum0 = __SMLAD(c20, k20, sum0); // 00 22
            sum0 = __SMLAD(c31, k31, sum0); // 11 33
            sum0 = __SMLABB(c64, k4, sum0);   // 44
            sum1 = __SMLATB(c20, k20, sum1); // 20
            sum1 = __SMLATB(c31, k31, sum1); // 31
            sum1 = __SMLABT(c64, k20, sum1); // 42
            sum1 = __SMLABT(c75, k31, sum1); // 53
            sum1 = __SMLATB(c64, k4, sum1); // 64
            cols_8b += input_w;
            k += 5;
        """, indent)
    content += chconv_mac_output(2, False, indent)
    return content


def chconv_k7x7_mac_setup(indent: int) -> str:
    return indent_lines(f"""
        int32_t k3210, k20, k31, k7654, k64, k75;
        int32_t c3210, c20, c31, c7654, c75, c64, c8;
    """, indent)


def chconv_k7x7_stride1_o2_mac(indent: int) -> str:
    content = chconv_mac_setup(1, 2, False, indent)
    content += indent_lines("const int8_t *k = ksrc;", indent)
    for i in range(0, 7):
        content += indent_lines(f"""
            // row {i}
            c3210 = read_int8x4(cols_8b);
            c20 = __SXTB16(c3210);
            c31 = __SXTB16_RORn(c3210, 8);
            c7654 = read_int8x4(cols_8b);
            c64 = __SXTB16(c7654);
            c75 = __SXTB16_RORn(c7654, 8);
            k3210 = read_int8x4(k);
            k20 = __SXTB16(k3210);
            k31 = __SXTB16_RORn(k3210, 8);
            k7654 = read_int8x4(k);
            k64 = __SXTB16(k7654);
            k75 = __SXTB16_RORn(k7654, 8);
            sum0 = __SMLAD(c20, k20, sum0); // 00 22
            sum0 = __SMLAD(c31, k31, sum0); // 11 33
            sum0 = __SMLAD(c64, k64, sum0); // 66 44
            sum0 = __SMLABB(c75, k75, sum0); // 55
            
            sum1 = __SMLAD(c31, k20, sum1); // 10 + 32
            sum1 = __SMLATB(c20, k31, sum1); // 21
            sum1 = __SMLABT(c64, k31, sum1); // 43
            sum1 = __SMLAD(c75, k64, sum1); //76, 54
            sum1 = __SMLATB(c64, k75, sum1); // 65
            cols_8b += input_w;
            k += 7;
        """, indent)
    content += chconv_mac_output(2, False, indent)
    return content


def chconv_k7x7_stride2_o2_mac(indent: int) -> str:
    content = chconv_mac_setup(1, 2, False, indent)
    content += indent_lines("const int8_t *k = ksrc;", indent)
    for i in range(0, 7):
        content += indent_lines(f"""
            // row {i}
            c3210 = read_int8x4(cols_8b);
            c20 = __SXTB16(c3210);
            c31 = __SXTB16_RORn(c3210, 8);
            c7654 = read_int8x4(cols_8b);
            c64 = __SXTB16(c7654);
            c75 = __SXTB16_RORn(c7654, 8);
            c8 = cols_8b[8];
            k3210 = read_int8x4(k);
            k20 = __SXTB16(k3210);
            k31 = __SXTB16_RORn(k3210, 8);
            k7654 = read_int8x4(k);
            k64 = __SXTB16(k7654);
            k75 = __SXTB16_RORn(k7654, 8);
            sum0 = __SMLAD(c20, k20, sum0); // 00 22
            sum0 = __SMLAD(c31, k31, sum0); // 11 33
            sum0 = __SMLAD(c64, k64, sum0); // 66 44
            sum0 = __SMLABB(c75, k75, sum0); // 55
            
            sum1 = __SMLATB(c20, k20, sum1); // 20
            sum1 = __SMLATB(c31, k31, sum1); // 31
            sum1 = __SMLABT(c64, k20, sum1); // 42
            sum1 = __SMLATT(c75, k31, sum1); // 53
            sum1 = __SMLATB(c64, k64, sum1); // 64
            sum1 = __SMLATB(c75, k75, sum1); // 75
            sum1 = __SMLABT( c8, k64, sum1); // 86
            cols_8b += input_w;
            k += 7;
        """, indent)
    content += chconv_mac_output(2, False, indent)
    return content


def chconv_generic_mac(kernel_size: int, stride: int, rev: bool, indent: int) -> str:
    content = chconv_mac_setup(stride, 1, rev, indent)
    for i in range(0, kernel_size * kernel_size):
        if i % kernel_size == 0 and i != 0:
            content += indent_lines(f"cols_8b += input_w;", indent)
        content += indent_lines(
            f"sum0 += cols_8b[{i%kernel_size}] * ksrc[{i}];", indent)
    content += chconv_mac_output(1, rev, indent)
    return content


def chconv_generic_content(kernel_size: int, stride: int, rev: bool, indent: int) -> str:
    content = chconv_setup(stride, rev, indent)
    content += indent_lines("for(int i = out_h; i > 0; i--){", indent)
    indent += 1
    content += indent_lines("for(int j = out_w; j > 0; j--){", indent)
    indent += 1
    content += chconv_generic_mac(kernel_size, stride, rev, indent)
    update_symbol = "-" if rev else "+"
    content += indent_lines(f"""
        }}
        input_col {update_symbol}= row_offset;""", indent)
    indent -= 1
    content += indent_lines("}", indent)
    return content


def chconv_preset_content(kernel_size: int, stride: int, rev: bool, indent: int) -> str:
    content = chconv_setup(stride, rev, indent)
    match kernel_size:
        case 7: content += chconv_k7x7_mac_setup(indent)
        case 5: content += chconv_k5x5_mac_setup(indent)
        case 3: content += chconv_k3x3_mac_setup(indent)
        case _: raise NotImplementedError()
    content += indent_lines("for(int i = out_h; i > 0; i--){", indent)
    indent += 1
    content += indent_lines("for(int j = out_w/2; j > 0; j--){", indent)
    indent += 1
    match kernel_size, stride:
        case 7, 2: content += chconv_k7x7_stride2_o2_mac(indent)
        case 7, 1: content += chconv_k7x7_stride1_o2_mac(indent)
        case 5, 2: content += chconv_k5x5_stride2_o2_mac(indent)
        case 5, 1: content += chconv_k5x5_stride1_o2_mac(indent)
        case 3, 2: content += chconv_k3x3_stride2_o2_mac(rev, indent)
        case 3, 1: content += chconv_k3x3_stride1_o2_mac(rev, indent)
        case _, _: raise NotImplementedError()
    indent -= 1
    content += indent_lines("}", indent)
    content += indent_lines("if(out_w % 2 != 0){", indent)
    indent += 1
    content += chconv_generic_mac(kernel_size, 1, rev, indent)
    indent -= 1
    update_symbol = "-" if rev else "+"
    content += indent_lines(f"""
        }}
        input_col {update_symbol}= row_offset;""", indent)
    indent -= 1
    content += indent_lines("}", indent)
    return content


def depconv_setup(model: Model, op: DepthConv2D, indent: int) -> str:
    input = model.tensors[op.input_idx]
    weight = model.tensors[op.weight_idx]
    output = model.tensors[op.output_idx]
    input_update = "1" if input.layout == DataLayout.HWC else "input_h * input_w"
    out_update = "1" if output.layout == DataLayout.HWC else "out_h * out_w"
    ch_offset = str(output.dim_c) if output.layout == DataLayout.HWC else "out_h * out_w"
    return indent_lines(f"""
        const int8_t pad_value = {-input.zero_point[0]};
        const int8_t out_offset = {output.zero_point[0]};
        const int8_t *weight = {weight_name(op.idx)};
        const float *scales = {scales_name(op.idx)};
        const int32_t *contrib = {contrib_name(op.idx)};
        const int input_h = {input.dim_h + 2 * input.prepad_h};
        const int input_w = {input.dim_w + 2 * input.prepad_h};
        const int input_ch = {input.dim_c};
        const int row_size = {input.dim_w + 2 * input.prepad_h + 2 * op.pad_w};
        const int out_h = {output.dim_h};
        const int out_w = {output.dim_w};
        const int input_update = {input_update};
        const int out_update = {out_update};
        const int ch_offset = {ch_offset};
        int8_t *out = output;
    """, indent)


def depconv_pad_buffer(pad_h: int, pad_w: int, indent: int) -> str:
    return indent_lines(f"""
        int8_t *pad_pos = buffer;
        memset(pad_pos, pad_value, row_size * {pad_h});
        pad_pos += row_size;
        for (int i = input_h; i > 0; i--) {{
            memset(pad_pos, pad_value, {pad_w});
            pad_pos += input_w + {pad_w};
            memset(pad_pos, pad_value, {pad_w});
            pad_pos += {pad_w};
        }}
        memset(pad_pos, pad_value, row_size * {pad_h});""", indent)


def depconv_loop_setup(indent: int) -> str:
    return indent_lines(f"""
        for(int c = input_ch; c > 0; c--) {{
           const int8_t *input_elem = input; """, indent)


def depconv_buffer_copy_hwc(pad_h: int, pad_w: int, indent: int) -> str:
    return indent_lines(f"""
        int8_t *buffer_elem = buffer + {pad_h} * row_size + {pad_w};
        for (int i = input_h; i > 0; i--) {{
            for (int j = input_w; j > 0; j--) {{
                *(buffer_elem++) = *input_elem;
                input_elem += input_ch;
            }}
            buffer_elem += 2 * {pad_w};
        }}""", indent)


def depconv_buffer_copy_chw(pad_h: int, pad_w: int, indent: int) -> str:
    return indent_lines(f"""
        int8_t *buffer_elem = buffer + {pad_h} * row_size + {pad_w};
        for (int i = input_h; i > 0; i--) {{
            memcpy(buffer_elem, input_elem, input_w);
            buffer_elem += row_size;
            input_elem += input_w;
        }}""", indent)


def depconv_buffer_copy(pad_h: int, pad_w: int, input_layout: DataLayout, indent: int) -> str:
    match input_layout:
        case DataLayout.CHW: return depconv_buffer_copy_chw(pad_h, pad_w, indent)
        case DataLayout.HWC: return depconv_buffer_copy_hwc(pad_h, pad_w, indent)
        case _: raise NotImplementedError(f"unknown data layout: {input_layout}")


def depconv_loop_body(input: Tensor, weight: Tensor, op: DepthConv2D, output_code: OutputCode, indent: int) -> str:
    input_str = "input"
    pad_input = op.pad_h != 0 or op.pad_w != 0
    content = ""
    if pad_input or input.layout != DataLayout.CHW:
        content += depconv_buffer_copy(op.pad_h, op.pad_w, input.layout, indent)
        input_str = "buffer"
    assert weight.dim_h == weight.dim_w
    assert op.stride_h == op.stride_w
    ch_conv =  output_code.add_ch_conv(weight.dim_h, op.stride_h, False)
    ch_conv_call = ch_conv.get_call(
        input_str, "out", "weight", 
        "*(scales++)", "*(contrib++)", "out_offset", 
        "row_size", "ch_offset", "out_w", "out_h")
    content += indent_lines(f"""
        {ch_conv_call};
        weight += {weight.dim_h * weight.dim_w};
        input += input_update;
        out += out_update;""", indent)
    return content