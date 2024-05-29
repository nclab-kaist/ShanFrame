from . import DataLayout, Operator, OperatorType, Model
from numpy import float64


class Conv2D(Operator):
    input_idx: float64 = float64(-1)
    weight_idx: float64 = float64(-1)
    pad_h: int = 0
    pad_w: int = 0
    stride_h: int = 1
    stride_w: int = 1
    bias_idx: float64 = float64(-1)
    io_overlap: bool = False
    buffer_size: int = 0
    buffer_addr: int = 0

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
    
    def min_buffer_size(self, model: Model) -> int:
        weight_tensor = model.tensors[self.weight_idx]
        # if is pointwise, no buffer needed
        if weight_tensor.dim_h == weight_tensor.dim_w == 1:
            return 0
        # minimum im2col buffer is one column (input channel number)
        input_tensor = model.tensors[self.input_idx]
        return weight_tensor.dim_h * weight_tensor.dim_w * input_tensor.dim_c
    
class DepthConv2D(Operator):
    input_idx: float64 = float64(-1)
    weight_idx: float64 = float64(-1)
    pad_h: int = 0
    pad_w: int = 0
    stride_h: int = 1
    stride_w: int = 1
    bias_idx: float64 = float64(-1)
    io_overlap: bool = False
    buffer_size: int = 0
    buffer_addr: int = 0

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

    def min_buffer_size(self, model: Model) -> int:
        input_tensor = model.tensors[self.input_idx]
        buffer_h = input_tensor.dim_h + 2 * input_tensor.prepad_h + 2 * self.pad_h
        buffer_w = input_tensor.dim_w + 2 * input_tensor.prepad_w + 2 * self.pad_w
        channel_size = buffer_h * buffer_w
        # if padding is fused, nothing we can do
        if self.pad_h != 0 or self.pad_w != 0:
            return channel_size
        # if input data layout is CHW, no data conversion buffer needed
        if input_tensor.layout == DataLayout.CHW:
            return 0
        return channel_size

class Add(Operator):
    input_idx: tuple[float64, float64]
    def __init__(self, input1_idx: float64, input2_idx: float64, output_idx: float64) -> None:
        input_idx_list = [input1_idx, input2_idx]
        super().__init__(input_idx_list, output_idx, OperatorType.ADD)
        self.input_idx = (input1_idx, input2_idx)

class Mul(Operator):
    input_idx: tuple[float64, float64]
    

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