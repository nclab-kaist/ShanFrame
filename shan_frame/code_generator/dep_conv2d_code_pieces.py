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
        int32_t sum0 = bias;
        int32_t sum1 = bias;
        
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
    content += indent_lines("for(int i = out_y; i > 0; i--){", indent)
    indent += 1
    content += indent_lines("for(int j = out_x; j > 0; j--){", indent)
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
    content += indent_lines("for(int i = out_y; i > 0; i--){", indent)
    indent += 1
    content += indent_lines("for(int j = out_x/2; j > 0; j--){", indent)
    indent += 1
    content += chconv_k3x3_stride1_o2_mac(indent)
    indent -= 1
    content += indent_lines("}", indent)
    content += indent_lines("if(out_x % 2 != 0){", indent)
    indent += 1
    content += chconv_generic_mac(3, 1, indent)
    indent -= 1
    content += indent_lines("""
        }
        input_col += 2;""", indent)
    indent -= 1
    content += indent_lines("}", indent)
    return content
