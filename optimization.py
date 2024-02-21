from abc import abstractmethod, ABC


class Optimization(ABC):
    @abstractmethod
    def estimate_flash_overhead(self) -> int:
        raise NotImplementedError("Optimization.estimate_flash_overhead")

    @abstractmethod
    def estimate_sram_overhead(self) -> int:
        raise NotImplementedError("Optimization.estimation_sram_overhead")

    @abstractmethod
    def estimate_latency_overhead(self) -> int:
        raise NotImplementedError("Optimization.estimate_latency_overhead")
