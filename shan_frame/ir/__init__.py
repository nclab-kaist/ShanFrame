from enum import Enum

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
    dim_h: int = -1
    dim_w: int = -1
    dim_c: int = -1
    src_op: int = -1
    dst_op: list[int] = []
    layout: DataLayout = DataLayout.UNKNOWN
    addr: int = 0
    data_type: DataType = DataType.INT8
    pad_h: int = -1
    pad_w: int = -1
    
    
class ConstArray:
    name: str
    dim_h: int = -1
    dim_w: int = -1
    dim_c: int = -1
    data_type: DataType
    data: list[float]


class Model:
    tensors: list[Tensor]
    operators: list[Operator]
