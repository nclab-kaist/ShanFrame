# from enum import Enum
from abc import ABC, abstractmethod


class Operand(ABC):

    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError("Operand.print")


class Expression(ABC):

    @abstractmethod
    def uses_operand(self, operand: Operand) -> bool:
        raise NotImplemented("Expression.use_operand")

    @abstractmethod
    def to_str(self, indent: int) -> str:
        raise NotImplemented("Expression.to_str")
