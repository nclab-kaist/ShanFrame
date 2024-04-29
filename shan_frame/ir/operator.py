from ir import Operator


class Conv2D(Operator):
    input_idx: int = -1
    output_idx: int = -1
    weight_idx: int = -1
    stride_h: int = -1
    stride_w: int = -1
    offset: int = -1
    bias_idx: int = -1
    scales_idx: int = -1

    
class DepthConv2D(Operator):
    input_idx: int = -1
    output_idx: int = -1
    weight_idx: int = -1
    stride_h: int = -1
    stride_w: int = -1
    offset: int = -1
    bias_idx: int = -1
    scales_idx: int = -1


class Add(Operator):
    input_idx: tuple[int, int] = (-1, -1)
    output_idx: int = -1


class Mul(Operator):
    input_idx: tuple[int, int] = (-1, -1)
    output_idx: int = -1
    
    