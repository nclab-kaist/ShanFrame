from typing import Self
from operand import ArrayOperand
from expression import ExpressionGroup


class Layer:
    name: str
    predecessors: list[Self]
    successors: list[Self]
    weights: list[ArrayOperand]
    content: ExpressionGroup


class Model:
    name: str
    layers: list[Layer]
    first_layer: Layer
    final_layer: Layer
