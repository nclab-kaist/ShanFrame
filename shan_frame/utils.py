import math
from typing import Any, Iterator
import numpy as np
from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP
from tflite import TensorType, BuiltinOperator

from ir import Tensor as IRTensor


class TFLiteTensorWrpper:
    def __init__(self, tensor_idx, tensor, buffer, qnn_params):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


def get_input_tensors(op: TFliteOP, model: TFliteModel) -> list[TFLiteTensorWrpper]:
    inputs = op.InputsAsNumpy()
    assert inputs != 0, f"no input found in {op}"
    return _get_wrapper_tensors(iter(inputs), model)


def get_output_tensors(op: TFliteOP, model: TFliteModel) -> list[TFLiteTensorWrpper]:
    inputs = op.OutputsAsNumpy()
    assert inputs != 0, f"no input found in {op}"
    return _get_wrapper_tensors(iter(inputs), model)


def _get_wrapper_tensors(tensor_index_list: Iterator[np.float64], model: TFliteModel):
    ret = []
    subgraph = model.Subgraphs(0)
    assert subgraph is not None, "No subgraph found"
    for idx in tensor_index_list:
        tensor = subgraph.Tensors(idx)
        assert tensor is not None, "No tensor found"
        buffer_idx = tensor.Buffer()
        buffer = model.Buffers(buffer_idx)

        tflite_qparams = tensor.Quantization()

        qparams_to_tensor_wrapper = None
        if tflite_qparams:
            scale = tflite_qparams.ScaleAsNumpy()
            if scale == 0:
                raise RuntimeError("No scale found")
            zero_point = tflite_qparams.ZeroPointAsNumpy()
            if isinstance(zero_point, np.ndarray):
                # Per-channel quantization
                if scale.size != 1 and zero_point.size != 1:
                    qparams_to_tensor_wrapper = {
                        "scale": scale, "zero_point": zero_point}
                # Per-tensor quantization
                elif scale.size == 1 and zero_point.size == 1:
                    qparams_to_tensor_wrapper = {"scale": float(
                        scale[0]), "zero_point": int(zero_point[0])}
                else:
                    raise NotImplementedError
            elif scale == zero_point == 0:
                pass
        else:
            raise NotImplementedError(
                "Quantization parameters not found in the model")

        ret.append(TFLiteTensorWrpper(
            idx, tensor, buffer, qparams_to_tensor_wrapper))
    return ret


def getTensorTypeStr(type):
    if TensorType.INT8 == type:
        return "int8"
    if TensorType.UINT8 == type:
        return "uint8"
    if TensorType.FLOAT32 == type:
        return "float32"


def get_np_from_wrapper(wrapper):
    if wrapper.tensor.Type() == TensorType.INT8:
        dtype = np.int8
    elif wrapper.tensor.Type() == TensorType.INT32:
        dtype = np.int32
    elif wrapper.tensor.Type() == TensorType.FLOAT32:
        # dtype = np.float32
        raise NotImplementedError("float point type is not supported")
    else:
        raise NotImplementedError(
            "Current implementation only supports int8 and int32")

    data = wrapper.buffer.DataAsNumpy()
    shape = wrapper.tensor.ShapeAsNumpy() if wrapper.tensor.ShapeLength() != 0 else []

    return np.frombuffer(data, dtype=dtype).reshape(shape)


def getMultiplierShift(effective_scale):
    significand = np.zeros(len(effective_scale), dtype="int32")
    shift = np.zeros(len(effective_scale), dtype="int32")

    for i, s in enumerate(effective_scale):
        if s == 0:
            significand[i] = 0
            shift[i] = 0
        else:
            sig, shi = math.frexp(s)
            sig = int(round(sig * 2**31))

            if sig == 2**31:
                sig /= 2
                shi += 1
            if shi < -31:
                shi = 0
                sig = 0

            significand[i] = sig
            shift[i] = shi

    return significand, shift


def getOpCodeStr(op, model: TFliteModel):
    op_code_list_idx = op.OpcodeIndex()
    op_code = model.OperatorCodes(op_code_list_idx)
    assert op_code is not None, "No op code found"
    op_code_id = op_code.DeprecatedBuiltinCode()

    def _build_str_map(obj):
        ret = {}
        for field_name in dir(obj):
            if not field_name.startswith("_"):
                field_value = getattr(obj, field_name)
                if isinstance(field_value, int):
                    ret[field_value] = field_name
        return ret

    builtin_op_code = _build_str_map(BuiltinOperator())

    return builtin_op_code[op_code_id]
