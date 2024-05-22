import math

import numpy as np

from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

from ..ir.operator import Pad
from ..ir import Tensor as IRTensor

from ..utils import (
    get_input_tensors,
    get_output_tensors,
    getOpCodeStr,
)

from ..ir import Model as IRModel


def parse_pad(op: TFliteOP, tflite_model: TFliteModel, ir_model: IRModel):
    # operator
    op_code_str = getOpCodeStr(op, tflite_model)
    assert op_code_str == "PAD"

    new_tensors: list[IRTensor] = []

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, tflite_model)
    new_tensors.extend(input_tensors)
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, tflite_model)
    new_tensors.extend(output_tensors)
    output_tensor_count = len(output_tensors)
    assert output_tensor_count == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # tensor types
    assert input_tensor.data.dtype == output_tensor.data.dtype, "tensor type not consistent"

    pad_op = Pad(
        input_tensor.tflite_tensor_idx,
        output_tensor.tflite_tensor_idx,
    )
    ir_model.add_tensors(new_tensors)
    ir_model.add_operator(pad_op)
