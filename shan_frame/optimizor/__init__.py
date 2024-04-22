from ir import IRModel


class Optimizer:
    def __init__(self, model: IRModel) -> None:
        pass

    def optimize(self, sram_scale: float = 1) -> tuple[IRModel, int]:
        '''sram_scale: allowed peak sram usage against minimum usage'''
        raise NotImplementedError("Optimizer.optimize()")
    
