import math

import numpy as np

from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Pool2DOptions import Pool2DOptions

from ..ir import Tensor as IRTensor
from ..ir.operator import AvgPool2D

from ..utils import (
    get_input_tensors,
    get_output_tensors,
    getOpCodeStr,
)

from ..ir import Model as IRModel


def parse_avgpool2d(op: TFliteOP, tflite_model: TFliteModel, ir_model: IRModel):
    # operator
    op_code_str = getOpCodeStr(op, tflite_model)
    assert op_code_str == "AVERAGE_POOL_2D"

    new_tensors: list[IRTensor] = []

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, tflite_model)
    new_tensors.extend(input_tensors)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 1, "input tensors length should be 1"
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, tflite_model)
    new_tensors.extend(output_tensors)
    output_tensor_count = len(output_tensors)
    assert output_tensor_count == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # tensor types
    assert input_tensor.data_type == output_tensor.data_type, "tensor type not consistent"

    # pool parameters
    assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
    op_options = op.BuiltinOptions()
    assert op_options is not None, "op has no options"
    pool2d_options = Pool2DOptions()
    pool2d_options.Init(op_options.Bytes, op_options.Pos)
    stride_h = pool2d_options.StrideH()
    stride_w = pool2d_options.StrideW()
    filter_h = pool2d_options.FilterHeight()
    filter_w = pool2d_options.FilterWidth()

    avgpool2d_op = AvgPool2D(
        input_tensor.tflite_tensor_idx, 
        output_tensor.tflite_tensor_idx,
        stride_h, stride_w,
        filter_h, filter_w
    )
    ir_model.add_tensors(new_tensors)
    ir_model.add_operator(avgpool2d_op)
