from ir import IROperator

class Conv2D(IROperator):
    input_idx: int
    output_idx: int
    weight_idx: int
    offset: int
    bias_idx: int
    scales_idx: int

    
class DepthConv2D(IROperator):
    input_idx: int
    output_idx: int
    weight_idx: int
    offset: int
    bias_idx: int
    scales_idx: int


class Add(IROperator):
    input_idx: tuple[int, int]
    output_idx: int

class Mul(IROperator):
    input_idx: tuple[int, int]
    output_idx: int
    
    