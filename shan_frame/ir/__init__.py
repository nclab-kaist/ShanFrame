from enum import Enum
import numpy as np
from typing import Iterable

class OperatorType(Enum):
    CONV_2D = 0
    DEPTH_CONV_2D = 1
    ADD = 2
    MUL = 3
    AVG_POOL_2D = 4
    PAD = 5
    RESHAPE = 6
    # To be added


class Operator:
    input_idx_list: list[np.float64]
    output_idx: np.float64
    op_type: OperatorType
    pad_output: bool

    def __init__(self, input_idx_list: list[np.float64], output_idx: np.float64, op_type: OperatorType) -> None:
        self.input_idx_list = input_idx_list
        self.output_idx = output_idx
        self.op_type = op_type

class DataLayout(Enum):
    UNKNOWN = 0
    HWC = 1
    CHW = 2


class DataType(Enum):
    INT8 = 0
    INT32 = 1
    FLOAT32 = 2


class Quantization(Enum):
    NO_QUANTIZATION = 0
    PER_CHANNEL = 1
    PER_TENSOR = 2

class Tensor:
    name: str
    tflite_tensor_idx: np.float64
    # shape
    dim_n: int = -1
    dim_h: int = -1
    dim_w: int = -1
    dim_c: int = -1
    # quant
    quant_type: Quantization
    scales: np.ndarray
    zero_point: np.ndarray
    # data
    data: np.ndarray
    # graph info
    src_op: int = -1
    dst_op: set[int]
    layout: DataLayout = DataLayout.HWC
    addr: int = 0
    # pre-padding
    prepad_h: int = 0
    prepad_w: int = 0
    def __init__(self) -> None:
        self.dst_op = set()


class Model:
    tensors: dict[np.float64, Tensor]
    operators: dict[int, Operator]
    _operator_counter = 0

    def __init__(self) -> None:
        self.tensors = {}
        self.operators = {}

    def add_tensors(self, tensors: Iterable[Tensor]):
        for tensor in tensors:
            idx = tensor.tflite_tensor_idx
            if self.tensors.get(idx) is None:
                self.tensors[idx] = tensor

    def add_operator(self, op: Operator):
        op_idx = self._operator_counter
        self._operator_counter += 1
        self.operators[op_idx] = op
        for input_idx in op.input_idx_list:
            input_tensor = self.tensors.get(input_idx)
            assert input_tensor is not None, f"input {int(input_idx)} of op {op_idx} does not exist"
            input_tensor.dst_op.add(op_idx)
        output_tensor = self.tensors.get(op.output_idx)
        assert output_tensor is not None, f"output {int(op.output_idx)} of op {op_idx} does not exist"
        output_tensor.src_op = op_idx

    def trim_operator(self):
        op_idx_list = list(self.operators.keys())
        op_idx_list.sort()
        op_new_idx_dict = {}
        
        for new_idx in range(0, len(op_idx_list)):
            old_idx = op_idx_list[new_idx]
            op_new_idx_dict[old_idx] = new_idx
            
        for tensor in self.tensors.values():
            if tensor.src_op >= 0:
                tensor.src_op = op_new_idx_dict[tensor.src_op]
            tensor.dst_op = {op_new_idx_dict[idx] for idx in tensor.dst_op}
            
        for old_idx, new_idx in op_new_idx_dict.items():
            self.operators[new_idx] = self.operators.pop(old_idx)
            
    def __str__(self) -> str:
        result = ""
        for idx, op in self.operators.items():
            result += f"{idx}: {op.op_type}\n"
        return result 