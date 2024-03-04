from typing import Self
from .operand import Array3DOperand
from .expression import ExpressionGroup


class Layer:
    name: str
    predecessors: list[Self]
    successors: list[Self]
    weights: list[Array3DOperand]
    buffers: list[Array3DOperand]
    content: ExpressionGroup


class Model:
    name: str
    layers: list[Layer]
    first_layer: Layer
    final_layer: Layer
