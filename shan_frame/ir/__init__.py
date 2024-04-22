from enum import Enum

class OperatorType(Enum):
    CONV_2D = 0
    DEPTH_CONV_2D = 1
    ADD = 2
    MUL = 3
    # To be added


class IROperator:
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
    dim_h: int
    dim_w: int
    dim_c: int
    src_op: int
    dst_op: list[int]
    layout: DataLayout
    addr: int
    data_type: DataType
    data: list[float]
    pad_h: int
    pad_w: int


class IRModel:
    tensors: list[Tensor]
    operators: list[IROperator]
