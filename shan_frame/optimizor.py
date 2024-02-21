from ir.model import Model
from optimization import get_optimizations, Optimization, OptimizationOption
from estimator import Estimator
from shan_frame import TargetArch

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
        current_usage = self.estimator.estimate_model(self.model)
        self.sram_current = current_usage.sram
        self.flash_current = current_usage.flash
        
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
        