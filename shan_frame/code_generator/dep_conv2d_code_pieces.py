from .conv2d_code_pieces import *
from .output_code import VecMulFunc
from .utils import indent_lines


def chconv_setup(indent: int) -> str:
    return indent_lines("""const int8_t *input_col = input;
        int8_t *out = output;""", indent)


def chconv_k3x3_mac_setup(indent: int) -> str:
    return indent_lines(f"""
        int32_t k3210, k20, k31, c3210, c20, c31;
        k3210 = arm_nn_read_q7x4(ksrc);
        k20 = __SXTB16(k3210);
        k31 = __SXTB16_RORn(k3210, 8);
        int32_t k7654, k64, k75, ka8;
        k7654 = arm_nn_read_q7x4(ksrc + 4);
        k64 = __SXTB16(k7654);
        k75 = __SXTB16_RORn(k7654, 8);
        ka8 = ksrc[8];""", indent)


def chconv_k3x3_stride1_o2_mac(indent: int) -> str:
    return indent_lines(f"""
        const int8_t *cols_8b = input_col;
        input_col += 2;
        int32_t sum0 = contrib;
        int32_t sum1 = contrib;
        
        c3210 = arm_nn_read_q7x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        sum0 = __SMLAD(c20, k20, sum0); // 0*0 + 2*2
        sum0 = __SMLABB(c31, k31, sum0); // += 1*1
        sum1 = __SMLAD(c31, k20, sum1); // 1*0 + 3*2
        sum1 = __SMLATB(c20, k31, sum1); // += 2*1
        cols_8b += row_size;
        
        c3210 = arm_nn_read_q7x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        sum0 = __SMLABT(c20, k31, sum0); // += 0 * 3
        sum0 = __SMLABB(c31, k64, sum0); // += 1 * 4
        sum0 = __SMLATB(c20, k75, sum0); // += 2 * 5
        sum1 = __SMLABT(c31, k31, sum1); // += 1 * 3
        sum1 = __SMLATB(c20, k64, sum1); // += 2 * 4
        sum1 = __SMLATB(c31, k75, sum1); // += 3 * 5
        cols_8b += row_size;

        c3210 = arm_nn_read_q7x4(cols_8b);
        c20 = __SXTB16(c3210);
        c31 = __SXTB16_RORn(c3210, 8);
        sum0 = __SMLABT(c20, k64, sum0); // += 0 * 6
        sum0 = __SMLABT(c31, k75, sum0); // += 1 * 7
        sum0 = __SMLATB(c20, ka8, sum0); // += 2 * 8
        sum1 = __SMLABT(c31, k64, sum1); // += 1 * 6
        sum1 = __SMLATT(c20, k75, sum1); // += 2 * 7
        sum1 = __SMLATB(c31, ka8, sum1); // += 3 * 8
        
        sum0 = (float)sum0 * scale;
        sum0 += out_offset;
        sum0 = MAX(sum0, -128);
        sum0 = MIN(sum0, 127);
        *out = sum0;
        out += ch_offset;        
        
        sum1 = (float)sum1 * scale;
        sum1 += out_offset;
        sum1 = MAX(sum1, -128);
        sum1 = MIN(sum1, 127);
        *out = sum1;
        out += ch_offset;""", indent)


def chconv_generic_mac(kernel_size: int, stride: int, indent: int) -> str:
    content = indent_lines(f"""
        const int8_t *cols_8b = input_col;
        input_col += {stride};
        int32_t sum = bias;
    """, indent)
    for i in range(0, kernel_size * kernel_size):
        if i % kernel_size == 0 and i != 0:
            content += indent_lines(f"cols_8b += row_size;", indent)
        content += indent_lines(
            f"sum += cols_8b[{i%kernel_size}] * ksrc[{i}];", indent)
    content += indent_lines("""
        sum = (float)sum * scale;
        sum += out_offset;
        sum = MAX(sum, -128);
        sum = MIN(sum, 127);
        *out = sum;
        out += ch_offset;""", indent)
    return content


def chconv_generic_content(kernel_size: int, stride: int, indent: int) -> str:
    content = chconv_setup(indent)
    content += indent_lines("for(int i = out_h; i > 0; i--){", indent)
    indent += 1
    content += indent_lines("for(int j = out_w; j > 0; j--){", indent)
    indent += 1
    content += chconv_generic_mac(kernel_size, stride, indent)
    content += indent_lines("""
        }
        input_col += row_offset;""", indent)
    indent -= 1
    content += indent_lines("}", indent)
    return content


def chconv_k3x3_stride1_content(indent: int) -> str:
    content = chconv_setup(indent)
    content += chconv_k3x3_mac_setup(indent)
    content += indent_lines("for(int i = out_h; i > 0; i--){", indent)
    indent += 1
    content += indent_lines("for(int j = out_w/2; j > 0; j--){", indent)
    indent += 1
    content += chconv_k3x3_stride1_o2_mac(indent)
    indent -= 1
    content += indent_lines("}", indent)
    content += indent_lines("if(out_w % 2 != 0){", indent)
    indent += 1
    content += chconv_generic_mac(3, 1, indent)
    indent -= 1
    content += indent_lines("""
        }
        input_col += 2;""", indent)
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
    ch_conv =  output_code.add_ch_conv(weight.dim_h, op.stride_h)
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