from shan_frame import compile_model_at

model_path = "./example/ad_small_int8.tflite"
output_dir = "./out"

sram_usage = compile_model_at(
    model_path, output_dir)

print(f"sram usage: {sram_usage}")
