from typing import Any, Self
from .definition import Operand, Expression, OperandType
from pprint import pprint

import shan_frame.utils as utils


class ElementOperand(Operand):
    name: str = ""
    type: OperandType
    max_value: float
    min_value: float
    source: Expression | None = None

    def __init__(self, type: OperandType, max_value: float, min_value: float, name: str = "") -> None:
        self.name = name
        self.max_value = max_value
        self.min_value = min_value
        self.type = type
        if len(name) == 0 and min_value != max_value:
            raise RuntimeError("Unnamed element is not literal")

    @classmethod
    def new_int_literal(cls, bit_len: int, value: int) -> Self:
        return cls(OperandType.new(True, bit_len), float(value), float(value))

    @classmethod
    def new_int(cls, bit_len: int, name: str) -> Self:
        type = OperandType.new(True, bit_len)
        return cls(type, type.max_value(), type.min_value(), name)

    @classmethod
    def new_float(cls, bit_len: int, name: str) -> Self:
        type = OperandType.new(False, bit_len)
        return cls(type, type.max_value(), type.min_value(), name)

    @classmethod
    def new_float_literal(cls, bit_len, value: float) -> Self:
        return cls(OperandType.new(False, bit_len), value, value)

    def has_known_value(self) -> bool:
        return self.max_value == self.min_value

    def _get_known_value(self) -> Any:
        value: float = self.max_value
        if value != self.min_value:
            return None
        if self.type.is_int():
            return int(value)
        return value

    def value_bit_length(self) -> int:
        if not self.type.is_int():
            return self.type.bit_len()
        max_value = int(self.max_value)
        min_value = int(self.min_value)
        return max(
            utils.signed_bit_length(max_value),
            utils.signed_bit_length(min_value)
        )

    def to_str(self) -> str:
        if len(self.name) != 0:
            return self.name
        value = self._get_known_value()
        if value is None:
            raise RuntimeError("Unnamed element is not literal")
        return str(value)


class Array3DOperand(Operand):
    name: str = ""
    type: OperandType
    #channel: int
    #x: int
    #y: int
    elements: list[list[list[ElementOperand]]]

    def __init__(self, name: str, type: OperandType, elements: list[list[list[ElementOperand]]]) -> None:
        self.type = type
        self.name = name
        #self.channel = channel
        #self.x = x
        #self.y = y
        self.elements = elements

    def to_str(self) -> str:
        if len(self.name) == 0:
            raise RuntimeError(f"Array is unnamed. {pprint(vars(self))}")
        return self.name

    @property
    def col_size(self):
        return len(self.elements[0])

    @property
    def row_size(self):
        return len(self.elements[0][0])

    @property
    def channel(self):
        return len(self.elements)
