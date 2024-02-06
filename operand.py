from abc import ABC, abstractmethod
from typing import Any, Self
from sys import float_info
from expression import Expression

import utils


class ElementType:
    is_int: bool
    bit_length: int

    def __init__(self, is_int: bool, bit_length: int) -> None:
        self.is_int = is_int
        self.bit_length = bit_length


class Operand(ABC):
    name: str | None

    @abstractmethod
    def print(self) -> str:
        raise NotImplementedError("Operand.print")


class ElementOperand(Operand):
    name: str | None = None
    type: ElementType
    max_value: float
    min_value: float
    source: Expression | None = None

    def __init__(self, is_int: bool, bit_length: int) -> None:
        self.type = ElementType(is_int, bit_length)
    
    @classmethod
    def new_int_value(cls, bit_len: int, value: int) -> Self:
        new_int = cls(True, bit_len)
        new_int.max_value = float(value)
        new_int.min_value = float(value)
        return new_int

    @classmethod
    def new_int(cls, bit_len: int) -> Self:
        new_int = cls(True, bit_len)
        value_max: int = (1 << bit_len) - 1
        new_int.max_value = float(value_max)
        new_int.min_value = float(-value_max - 1)
        return new_int

    @classmethod
    def new_float(cls, bit_len: int, value: float | None = None) -> Self:
        new_float = cls(False, bit_len)
        if value == None:
            new_float.max_value = float_info.max
            new_float.min_value = float_info.min
        else:
            new_float.max_value = value
            new_float.min_value = value
        return new_float

    def has_known_value(self) -> bool:
        return self.max_value == self.min_value

    def _get_known_value(self) -> Any:
        value: float = self.max_value
        if value != self.min_value:
            return None
        if self.type.is_int:
            return int(value)
        return value

    def value_bit_length(self) -> int:
        if self.type.is_int:
            return self.type.bit_length
        max_value = int(self.max_value)
        min_value = int(self.min_value)
        return max(
            utils.signed_bit_length(max_value),
            utils.signed_bit_length(min_value)
        )

    def print(self) -> str:
        value = self._get_known_value()
        if value is None:
            return str(self.name)
        else:
            return str(value)


class ArrayOperand(Operand):
    name: str
    type: ElementType
    size: int

    def __init__(self, name: str, type: ElementType, size: int) -> None:
        self.type = type
        self.size = size
        self.name = name

    def print(self) -> str:
        return self.name
