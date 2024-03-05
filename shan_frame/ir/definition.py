from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Self


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


class OperandType(IntEnum):
    VOID = 0
    INT1 = 1
    INT8 = 8
    INT16 = 16
    INT32 = 32
    INT64 = 64
    FLOAT32 = -32
    FLOAT64 = -64

    def to_str(self) -> str:
        match self:
            case OperandType.VOID:
                return "void"
            case OperandType.INT1:
                return "bool"
            case OperandType.INT8:
                return "int8_t"
            case OperandType.INT16:
                return "int16_t"
            case OperandType.INT32:
                return "int32_t"
            case OperandType.INT64:
                return "int64_t"
            case OperandType.FLOAT32:
                return "float"
            case OperandType.FLOAT64:
                return "double"

    def bit_len(self) -> int:
        return abs(int(self))

    def is_int(self) -> bool:
        return int(self) > 0

    @classmethod
    def new(cls, is_int: bool, bit_len: int) -> Self:
        value = bit_len
        if not is_int:
            value = -value
        return cls(value)
                
                