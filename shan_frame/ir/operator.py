from ..ir import Operator, OperatorType
from numpy import float64


class Conv2D(Operator):
    input_idx: float64 = float64(-1)
    weight_idx: float64 = float64(-1)
    pad_h: int = 0
    pad_w: int = 0
    stride_h: int = 1
    stride_w: int = 1
    bias_idx: float64 = float64(-1)

    def __init__(self, input_idx: float64, weight_idx: float64, output_idx: float64, pad_h: int = 0, pad_w: int = 0, stride_h: int = 0, stride_w: int = 0, bias_idx: float64 = float64(-1)) -> None:
        input_idx_list = [input_idx, weight_idx]
        if bias_idx >= 0:
            input_idx_list.append(bias_idx)
        super().__init__(input_idx_list, output_idx, OperatorType.CONV_2D)
        self.input_idx = input_idx
        self.weight_idx = weight_idx
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.bias_idx = bias_idx
    
class DepthConv2D(Operator):
    input_idx: float64 = float64(-1)
    weight_idx: float64 = float64(-1)
    pad_h: int = 0
    pad_w: int = 0
    stride_h: int = 1
    stride_w: int = 1
    bias_idx: float64 = float64(-1)

    def __init__(self, input_idx: float64, weight_idx: float64, output_idx: float64, pad_h: int = 0, pad_w: int = 0, stride_h: int = 0, stride_w: int = 0, bias_idx: float64 = float64(-1)) -> None:
        input_idx_list = [input_idx, weight_idx]
        if bias_idx >= 0:
            input_idx_list.append(bias_idx)
        super().__init__(input_idx_list, output_idx, OperatorType.DEPTH_CONV_2D)
        self.input_idx = input_idx
        self.weight_idx = weight_idx
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.bias_idx = bias_idx


class Add(Operator):
    input_idx: tuple[float64, float64]
    def __init__(self, input1_idx: float64, input2_idx: float64, output_idx: float64) -> None:
        input_idx_list = [input1_idx, input2_idx]
        super().__init__(input_idx_list, output_idx, OperatorType.ADD)
        self.input_idx = (input1_idx, input2_idx)

class Mul(Operator):
    input_idx: tuple[float, float] = (-1, -1)
    

class AvgPool2D(Operator):
    input_idx: float64 = float64(-1)
    stride_h: int
    stride_w: int
    filter_h: int
    filter_w: int

    def __init__(self, input_idx: float64, output_idx: float64, stride_h: int, stride_w: int, filter_h: int, filter_w: int) -> None:
        input_idx_list = [input_idx]
        super().__init__(input_idx_list, output_idx, OperatorType.AVG_POOL_2D)
        self.input_idx = input_idx
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.filter_h = filter_h
        self.filter_w = filter_w


class Pad(Operator):
    def __init__(self, input_idx: float64, output_idx: float64) -> None:
        input_idx_list = [input_idx]
        super().__init__(input_idx_list, output_idx, OperatorType.PAD)

class Reshape(Operator):
    def __init__(self, input_idx: float64, output_idx: float64) -> None:
        input_idx_list = [input_idx]
        super().__init__(input_idx_list, output_idx, OperatorType.RESHAPE)