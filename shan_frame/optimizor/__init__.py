from ..ir import Model


class Optimizer:
    def __init__(self, model: Model) -> None:
        pass

    def optimize(self, sram_scale: float = 1) -> tuple[Model, int]:
        '''sram_scale: allowed peak sram usage against minimum usage'''
        raise NotImplementedError("Optimizer.optimize()")
    
