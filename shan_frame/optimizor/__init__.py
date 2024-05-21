
from ..ir import Model
from ..ir.operator import Conv2D, DepthConv2D
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
        
        # iterative io overlap optimization
        for op in self.model.operators.values():
            if not (isinstance(op, Conv2D) or isinstance(op, DepthConv2D)):
                continue
            if op.io_overlap:
                op.io_overlap = False
                current_peak_mem = self.mem_scheduler.schedule(self.model)
                if current_peak_mem > self.min_peak_mem_usage * sram_scale:
                    op.io_overlap = True

        final_peak_mem = self.mem_scheduler.schedule(self.model)
        visualize_memory(self.model)
        
        return (self.model, final_peak_mem)
    
            