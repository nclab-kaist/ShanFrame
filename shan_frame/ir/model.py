from typing import Self
from shan_frame.ir.operand import ArrayOperand
from shan_frame.ir.expression import ExpressionGroup


class Layer:
    name: str
    predecessors: list[Self]
    successors: list[Self]
    weights: list[ArrayOperand]
    buffers: list[ArrayOperand]
    content: ExpressionGroup


class Model:
    name: str
    layers: list[Layer]
    first_layer: Layer
    final_layer: Layer
