from ir import Operator


class Conv2D(Operator):
    input_idx: float = -1
    output_idx: float = -1
    weight_idx: float = -1
    pad_h: int = -1
    pad_w: int = -1
    stride_h: int = -1
    stride_w: int = -1
    offset: int = -1
    bias_idx: float = -1
    scales_idx: float = -1

    
class DepthConv2D(Operator):
    input_idx: float = -1
    output_idx: float = -1
    weight_idx: float = -1
    pad_h: int = -1
    pad_w: int = -1
    stride_h: int = -1
    stride_w: int = -1
    offset: int = -1
    bias_idx: float = -1
    scales_idx: float = -1


class Add(Operator):
    input_idx: tuple[float, float] = (-1, -1)
    output_idx: float = -1


class Mul(Operator):
    input_idx: tuple[float, float] = (-1, -1)
    output_idx: float = -1
    
    