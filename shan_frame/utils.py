import math
from typing import Any, Iterator
import numpy as np
from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP
from tflite import TensorType, BuiltinOperator

from .ir import DataLayout, Quantization, Tensor as IRTensor, Model as IRModel, OperatorType
from .ir.operator import Conv2D, DepthConv2D

class TFLiteTensorWrpper:
    def __init__(self, tensor_idx, tensor, buffer, qnn_params):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


def get_input_tensors(op: TFliteOP, model: TFliteModel) -> list[IRTensor]:
    inputs = op.InputsAsNumpy()
    assert isinstance(inputs, np.ndarray), f"no input found in {op}"
    return _get_tensors(iter(inputs), model)


def get_output_tensors(op: TFliteOP, model: TFliteModel) -> list[IRTensor]:
    inputs = op.OutputsAsNumpy()
    assert isinstance(inputs, np.ndarray), f"no output found in {op}"
    return _get_tensors(iter(inputs), model)


def _get_tensors(tensor_index_list: Iterator[np.float64], model: TFliteModel) -> list[IRTensor]:
    ret = []
    subgraph = model.Subgraphs(0)
    assert subgraph is not None, "No subgraph found"
    for idx in tensor_index_list:
        ir_tensor = IRTensor()
        ir_tensor.tflite_tensor_idx = idx
        tensor = subgraph.Tensors(idx)

        assert tensor is not None, f"Tensor at idx {idx} found"
        # determine shape
        tensor_shape = tensor.ShapeAsNumpy()
        assert isinstance(
            tensor_shape, np.ndarray), f"Tensor at idx {idx} has no shape"
        match tensor_shape.size:
            case 4:
                ir_tensor.dim_n, ir_tensor.dim_h, ir_tensor.dim_w, ir_tensor.dim_c = tensor_shape
            case 2:
                ir_tensor.dim_n, ir_tensor.dim_c = 1, 1
                ir_tensor.dim_h, ir_tensor.dim_w = tensor_shape
            case 1:
                ir_tensor.dim_n, ir_tensor.dim_h, ir_tensor.dim_w = 1, 1, 1
                ir_tensor.dim_c = tensor_shape[0]
            case _:
                raise RuntimeError(
                    f"unsupported tensor shape dimension: {tensor_shape.size}")

        # determine data
        buffer_idx = tensor.Buffer()
        buffer = model.Buffers(buffer_idx)
        match tensor.Type():
            case TensorType.INT8:
                np_type = np.int8
            case TensorType.INT32:
                np_type = np.int32
            case TensorType.FLOAT32:
                np_type = np.float32
            case _:
                raise NotImplementedError(f"unsupported tensor type {tensor.Type()}")        
        data = np.ndarray([], np_type)
        if buffer is not None:
            data_tmp = buffer.DataAsNumpy()
            if isinstance(data_tmp, np.ndarray):
                data = np.frombuffer(data_tmp, dtype=np_type)
        ir_tensor.data = data

        # determine quantization
        tflite_qparams = tensor.Quantization()
        if tflite_qparams is None:
            # TODO: support floating-point operators with no quantization
            raise NotImplementedError("Quantization parameters not found")
        scale = tflite_qparams.ScaleAsNumpy()
        zero_point = tflite_qparams.ZeroPointAsNumpy()
        if isinstance(scale, np.ndarray) and isinstance(zero_point, np.ndarray):
            if scale.size != 1 and zero_point.size != 1:
                ir_tensor.quant_type = Quantization.PER_CHANNEL
            elif scale.size == 1 and zero_point.size == 1:
                ir_tensor.quant_type = Quantization.PER_TENSOR
            else:
                raise NotImplementedError("Unsupported quantization setup")
            ir_tensor.scales = scale
            ir_tensor.zero_point = zero_point
        else:
            ir_tensor.quant_type = Quantization.NO_QUANTIZATION
            ir_tensor.scales = np.ndarray([0], np.float64)
            ir_tensor.zero_point = np.ndarray([0], np.float64)
        ret.append(ir_tensor)
    return ret


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


class Rect:
    idx: np.float64
    height: int
    width: int
    start: int
    addr: int

    def __init__(self, idx: np.float64, height: int, weight: int, start: int, addr: int) -> None:
        self.idx = idx
        self.height = height
        self.width = weight
        self.start = start
        self.addr = addr

    def __str__(self) -> str:
        return f"{{h: {self.height}, w: {self.width}, ({self.start}, {self.addr})}}"


def get_rect(model: IRModel) -> list[Rect]:
    rect_list = []
    op_idx_list = list(model.operators.keys())
    op_idx_list.sort()
    assert op_idx_list[0] == 0 and op_idx_list[-1] == len(
        op_idx_list) - 1, "input model is not trimmed"
    for op_idx in op_idx_list:
        op = model.operators[op_idx]
        tensor_idx = op.output_idx
        tensor = model.tensors[tensor_idx]
        tensor_size = tensor.mem_size()
        # determine lifetime
        tensor_lifetime = 1
        for dst_op_idx in tensor.dst_op:
            lifetime = dst_op_idx - op_idx + 1
            dst_op = model.operators[dst_op_idx]
            if (isinstance(dst_op, DepthConv2D) or isinstance(dst_op, Conv2D)) and dst_op.io_overlap:
                lifetime = dst_op_idx - op_idx
            if tensor_lifetime < lifetime:
                tensor_lifetime = lifetime
        rect = Rect(tensor_idx, tensor_size,
                    tensor_lifetime, op_idx, tensor.addr)
        rect_list.append(rect)
    return rect_list


def get_align_groups(model: IRModel) -> list[tuple[set[np.float64], int, int]]:
    align_groups: list[tuple[set[np.float64], int, int]] = []
    for op in model.operators.values():
        if not isinstance(op, DepthConv2D):
            continue
        if not op.io_overlap:
            continue
        input = model.tensors[op.input_idx]
        output = model.tensors[op.output_idx]
        if not input.layout == output.layout == DataLayout.HWC:
            continue
        input_in_group = False
        for group in align_groups:
            if op.input_idx in group:
                input_in_group = True
                group[0].add(op.output_idx)
                break
        if not input_in_group:
            align_groups.append(({op.input_idx, op.output_idx}, input.dim_c, -1))
    return align_groups


def get_buf_rect(model: IRModel) -> list[Rect]:
    rect_list = []
    op_idx_list = list(model.operators.keys())
    op_idx_list.sort()
    assert op_idx_list[0] == 0 and op_idx_list[-1] == len(
        op_idx_list) - 1, "input model is not trimmed"
    for op_idx in op_idx_list:
        op = model.operators[op_idx]
        # if a buffer is required, generate rect
        if not (isinstance(op, DepthConv2D) or isinstance(op, Conv2D)):
            continue
        min_buffer_size = op.min_buffer_size(model)
        if min_buffer_size == 0:
            continue
        # buffer is required
        buf_start = 0 if op_idx == 0 else op_idx - 1
        buf_lifetime = op_idx + 1 - buf_start
        buffer_rect = Rect(np.float64(op_idx), min_buffer_size,
                           buf_lifetime, buf_start, 0)
        rect_list.append(buffer_rect)
    return rect_list
