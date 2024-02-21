from typing import Any, Self
from abc import abstractmethod, ABC
from ir.model import Model


class OptimizationOption(ABC):
    @abstractmethod
    def estimate_flash_overhead(self) -> int:
        pass

    @abstractmethod
    def estimate_sram_overhead(self) -> int:
        pass

    @abstractmethod
    def estimate_latency_overhead(self) -> int:
        pass

    @abstractmethod
    def do_optimization(self, model: Model) -> Model:
        pass


class Optimization(ABC):
    @abstractmethod
    def get_optimization_options(self, model: Model) -> list[OptimizationOption]:
        pass


def get_optimizations() -> list[Optimization]:
    return []
