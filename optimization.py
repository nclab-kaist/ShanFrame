from typing import Any, Self
from abc import abstractclassmethod, ABCMeta


class Optimization(metaClass=ABCMeta):
    @abstractclassmethod
    def estimate_flash_overhead(self) -> int:
        pass

    @abstractclassmethod
    def estimate_sram_overhead(self) -> int:
        pass

    @abstractclassmethod
    def estimate_latency_overhead(self) -> int:
        pass
