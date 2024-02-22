from shan_frame import compile_model_at, TargetArch

model_path = "./example/ad_small_int8.tflite"
output_dir = "./out"
target_arch = TargetArch.CORTEX_M7
sram_limit = 1024
flash_limit = 1024

(sram_usage, flash_usage) = compile_model_at(
    model_path, output_dir, target_arch, sram_limit, flash_limit)

print(f"sram usage: {sram_usage}, flash_usage: {flash_usage}")
