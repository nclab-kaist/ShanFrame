from shan_frame.ir.model import Model
from shan_frame.optimization import get_optimizations, Optimization, OptimizationOption
from shan_frame.estimator import Estimator
from shan_frame.target_arch import TargetArch

class Optimizor:
    sram_current: int
    sram_limit: int
    flash_current: int
    flash_limit: int
    model: Model
    estimator: Estimator
    optimizations: list[Optimization]
    options: list[OptimizationOption]
    def __init__(self, sram_limit: int, flash_limit: int, model: Model, output_dir: str, target_arch: TargetArch):
        self.sram_limit = sram_limit
        self.flash_limit = flash_limit
        self.model = model
        self.optimizations = get_optimizations(target_arch)
        self.options = list()
        self.estimator = Estimator(output_dir)
        (self.sram_current, self.flash_current) = self.estimator.estimate_model(self.model)
        
    def optimize(self) -> None:
        while self._optimize_one_round():
            pass
        
    def _pick_option(self) -> OptimizationOption:
        raise NotImplementedError("Optimizor._pick_option")
    
    def _optimize_one_round(self) -> bool:
        # get all options
        for optimization in self.optimizations:
            self.options += optimization.get_optimization_options(self.model)
        picked_opt = self._pick_option()
        new_model = picked_opt.do_optimization(self.model)
        estimate_result = self.estimator.estimate_model(new_model)
        raise NotImplementedError("Optimizor._optimizor_one_round")
        