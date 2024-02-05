from typing import Any, Self
from abc import abstractclassmethod, ABCMeta
from model import Model


class OptimizationOption(metaClass=ABCMeta):
    @abstractclassmethod
    def estimate_flash_overhead(self) -> int:
        pass

    @abstractclassmethod
    def estimate_sram_overhead(self) -> int:
        pass

    @abstractclassmethod
    def estimate_latency_overhead(self) -> int:
        pass

    @abstractclassmethod
    def do_optimization(self, model: Model) -> Model:
        pass


class Optimization(ABCMeta):
    @abstractclassmethod
    def get_optimization_options(self, model: Model) -> list[OptimizationOption]:
        pass

def get_optimizations() -> list[Optimization]:
    return []