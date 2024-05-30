from .model_parser import ModelParser
from .optimizor import Optimizer
from .code_generator import CodeGenerator

def compile_model_at(
    model_path: str,
    output_dir: str,
) -> int:
    model = ModelParser(model_path).parse_model()
    
    optimizer = Optimizer(model)
    (model, sram_usage)= optimizer.optimize(1.0)
    
    code_generator = CodeGenerator(output_dir)
    code_generator.generate(model, sram_usage)
    
    return sram_usage
