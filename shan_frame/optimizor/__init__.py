
from ..ir import Model
from .mem_scheduler import MemoryScheduler
from .visualizer import visualize_memory

class Optimizer:
    mem_scheduler: MemoryScheduler
    model: Model
    min_peak_mem_usage: int
    def __init__(self, model: Model) -> None:
        self.mem_scheduler = MemoryScheduler()
        self.model = model
        self.min_peak_mem_usage = self.mem_scheduler.schedule(model)
        pass

    def optimize(self, sram_scale: float = 1) -> tuple[Model, int]:
        '''sram_scale: allowed peak sram usage against minimum usage'''
        
        # TODO: iterative optimization
        
        visualize_memory(self.model)
        
        raise NotImplementedError("Optimizer.optimize()")
    
            