from abc import abstractmethod, ABC


class Optimization(ABC):
    @abstractmethod
    def estimate_flash_overhead(self) -> int:
        pass

    @abstractmethod
    def estimate_sram_overhead(self) -> int:
        pass

    @abstractmethod
    def estimate_latency_overhead(self) -> int:
        pass
