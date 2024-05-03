from enum import Enum
import numpy as np

class OperatorType(Enum):
    CONV_2D = 0
    DEPTH_CONV_2D = 1
    ADD = 2
    MUL = 3
    # To be added


class Operator:
    op_type: OperatorType
    pad_output: bool


class DataLayout(Enum):
    UNKNOWN = 0
    HWC = 1
    CHW = 2


class DataType(Enum):
    INT8 = 0
    INT32 = 1
    FLOAT32 = 2


class Tensor:
    name: str
    tflite_tensor_idx: float = -1
    dim_h: int = -1
    dim_w: int = -1
    dim_c: int = -1
    src_op: int = -1
    dst_op: list[int] = []
    layout: DataLayout = DataLayout.HWC
    addr: int = 0
    data_type: DataType = DataType.INT8
    data: np.ndarray
    prepad_h: int = -1
    prepad_w: int = -1


class Model:
    tensors: dict[float, Tensor]
    operators: list[Operator]
