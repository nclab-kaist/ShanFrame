
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
        
        # iterative optimization
        for op_idx, op in self.model.operators.items():
            if not (isinstance(op, Conv2D) or isinstance(op, DepthConv2D)):
                continue

            # try to de-overlap input and output
            if op.io_overlap:
                op.io_overlap = False
                current_peak_mem = self.mem_scheduler.schedule(self.model)
                if current_peak_mem > self.min_peak_mem_usage * sram_scale:
                    op.io_overlap = True

            # try to pre-pad input
            # XXX: This is assuming pre-padding has no interference with overlapping
            if op_idx != 0 and (op.pad_h != 0 or op.pad_w != 0):
                input_tensor = self.model.tensors[op.input_idx]
                input_tensor.prepad_h, input_tensor.prepad_w = op.pad_h, op.pad_w
                op.pad_h, op.pad_w = 0, 0
                current_peak_mem = self.mem_scheduler.schedule(self.model)
                if current_peak_mem > self.min_peak_mem_usage * sram_scale:
                    op.pad_h, op.pad_w = input_tensor.prepad_h, input_tensor.prepad_w
                    input_tensor.prepad_h, input_tensor.prepad_w = 0, 0


        final_peak_mem = self.mem_scheduler.schedule(self.model)
        visualize_memory(self.model)
        
        return (self.model, final_peak_mem)
    
            