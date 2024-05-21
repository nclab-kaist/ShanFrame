import math

import numpy as np

from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP

from ..ir import Tensor as IRTensor
from ..ir.operator import Add

from ..utils import (
    get_input_tensors,
    get_output_tensors,
    getOpCodeStr,
)

from ..ir import Model as IRModel


def parse_add(op: TFliteOP, tflite_model: TFliteModel, ir_model: IRModel):
    # operator
    op_code_str = getOpCodeStr(op, tflite_model)
    assert op_code_str == "ADD"

    new_tensors: list[IRTensor] = []

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, tflite_model)
    new_tensors.extend(input_tensors)
    assert len(input_tensors) == 2, "input should be 2 tensors"

    input1_tensor = input_tensors[0]
    input2_tensor = input_tensors[1]

    output_tensors = get_output_tensors(op, tflite_model)
    new_tensors.extend(output_tensors)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # tensor types
    assert input1_tensor.data_type == input2_tensor.data_type, "input tensor types not consistent"
    assert input1_tensor.data_type == output_tensor.data_type, "output tensor type not consistent"

    add_op = Add(
        input1_tensor.tflite_tensor_idx,
        input2_tensor.tflite_tensor_idx,
        output_tensor.tflite_tensor_idx
    )
    ir_model.add_tensors(new_tensors)
    ir_model.add_operator(add_op)
