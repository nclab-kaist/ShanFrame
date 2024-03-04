from typing import Self
from .operand import Array3DOperand
from .expression import ExpressionGroup
import attrs

@attrs.define() # If we want to make it immutable, set frozen=True
class Layer:
    name: str
    predecessors: list[Self]
    successors: list[Self]
    weights: list[Array3DOperand]
    buffers: list[Array3DOperand]
    content: ExpressionGroup

@attrs.define()
class Model:
    name: str
    layers: list[Layer]
    #first_layer: Layer
    #final_layer: Layer

    @property
    def first_layer(self)->Layer:
        if len(self.layers) == 0:
            raise ValueError("no layer inside")
        return self.layers[0]
    
    @property
    def final_layer(self)->Layer:
        if len(self.layers) == 0:
            raise ValueError("no layer inside")
        return self.layers[-1] 