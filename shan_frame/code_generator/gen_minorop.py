def generate_minor_declare() -> str:
    return """
#include <stdint.h>
void elementwise_add(int size, 
        const int8_t* input1_data, const float input1_scale, const float input1_zero,
        const int8_t* input2_data, const float input2_scale, const float input2_zero, 
        int8_t* output_data, const float output_scale, const float zero_y);
void avg_pooling(
        const int8_t* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
        int8_t* output, const uint16_t output_h, const uint16_t output_w,
        const uint16_t sample_h, const uint16_t sample_w);
"""

def generate_minor_def() -> str:
    return """
#include <stdint.h>
#include <math.h>

void elementwise_add(int size, 
        const int8_t* input1_data, const float input1_scale, const float input1_zero,
        const int8_t* input2_data, const float input2_scale, const float input2_zero, 
        int8_t* output_data, const float output_scale, const float zero_y) 
{
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = MAX(clamped_output, -128);
      clamped_output = MIN(clamped_output, 127);
      output_data[i] = (int8_t)(clamped_output);
  }
}

void avg_pooling(
        const int8_t* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
        int8_t* output, const uint16_t output_h, const uint16_t output_w,
        const uint16_t sample_h, const uint16_t sample_w)
{
	int h, w, c;
	int sh, sw;
	const int divider_half = ((sample_h * sample_w) / 2);
	for(c = 0; c < input_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				int avg = 0;

				for(sh = 0; sh < sample_h; sh++){
					int height = sh + h * sample_h;
					for(sw = 0; sw < sample_w; sw++){
						int width = sw + w * sample_w;
						avg += input[(width + height * input_w) * input_c + c];
					}
				}

				// for rounded div
				if (avg > 0)
					avg += divider_half;
				else
					avg -= divider_half;

				int out = avg / (sample_h * sample_w);
				out = MAX(out, -128);
				out = MIN(out, 127);
				output[(w + h * output_w) * input_c + c] = out;
			}
		}
	}
}
"""