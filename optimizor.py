from ir.model import Model
from optimization import get_optimizations, Optimization, OptimizationOption
from estimator import estimate_model

class Optimizor:
    sram_current: int
    sram_limit: int
    flash_current: int
    flash_limit: int
    model: Model
    optimizations: list[Optimization]
    options: list[OptimizationOption]
    def __init__(self, sram_limit: int, flash_limit: int, model: Model):
        self.sram_limit = sram_limit
        self.flash_limit = flash_limit
        self.model = model
        self.optimizations = get_optimizations()
        self.options = list()
        current_usage = estimate_model(self.model)
        self.sram_current = current_usage.sram
        self.flash_current = current_usage.flash
        
    def pick_option(self) -> OptimizationOption:
        # TODO: implement option picking policy
        raise NotImplementedError("Optimizor.pick_option")
    
    def optimize(self) -> bool:
        # get all options
        for optimization in self.optimizations:
            self.options += optimization.get_optimization_options(self.model)
        picked_opt = self.pick_option()
        new_model = picked_opt.do_optimization(self.model)
        estimate_result = estimate_model(new_model)
        raise NotImplementedError("Optimizor.optimizor")
        