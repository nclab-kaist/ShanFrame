from shan_frame.target_arch import TargetArch
from shan_frame.ir import IRGenerator
from shan_frame.optimizor import Optimizor
from shan_frame.code_generator import generate_code


def compile_model_at(
    model_path: str,
    output_dir: str,
    target_arch: TargetArch,
    sram_limit: int,
    flash_limit: int
) -> tuple[int, int]:
    model = IRGenerator(model_path).parse_mdoel()
    optimizor = Optimizor(sram_limit, flash_limit,
                          model, output_dir, target_arch)
    optimizor.optimize()
    sram_usage = optimizor.sram_current
    flash_usage = optimizor.flash_current
    generate_code(output_dir, optimizor.model)
    return (sram_usage, flash_usage)
