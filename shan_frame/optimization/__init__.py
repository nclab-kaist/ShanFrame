from typing import Any, Self
from abc import abstractmethod, ABC
from ir.model import Model
from shan_frame import TargetArch


class OptimizationOption(ABC):
    @abstractmethod
    def estimate_flash_overhead(self) -> int:
        raise NotImplementedError("OptimizationOption.estimate_flash_overhead")

    @abstractmethod
    def estimate_sram_overhead(self) -> int:
        raise NotImplementedError(
            "OptimizationOption.estimation_sram_overhead")

    @abstractmethod
    def estimate_latency_overhead(self) -> int:
        raise NotImplementedError(
            "OptimizationOption.estimate_latency_overhead")

    @abstractmethod
    def do_optimization(self, model: Model) -> Model:
        raise NotImplementedError("OptimizationOption.do_optimization")


class Optimization(ABC):
    @abstractmethod
    def get_optimization_options(self, model: Model) -> list[OptimizationOption]:
        raise NotImplementedError("Optimization.get_optimization_options")


def get_optimizations(target_arch: TargetArch) -> list[Optimization]:
    raise NotImplementedError("optimization.get_optimizations")
