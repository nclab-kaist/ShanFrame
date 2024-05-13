import math

import numpy as np 

from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

from ..ir.operator import Conv2D, DepthConv2D
from ..ir import Tensor as IRTensor

from ..utils import (
    get_input_tensors,
    get_output_tensors,
    getOpCodeStr,
)

from ..ir import Model as IRModel

def parse_conv2d(op: TFliteOP, tflite_model: TFliteModel, ir_model: IRModel):
    # operator
    op_code_str = getOpCodeStr(op, tflite_model)
    assert op_code_str == "CONV_2D" or op_code_str == "DEPTHWISE_CONV_2D", f"unsupported input op to parse_conv2d(): {op_code_str}"
    is_conv2d = (op_code_str == "CONV_2D")
    
    new_tensors: list[IRTensor] = []
    
    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, tflite_model)
    new_tensors.extend(input_tensors)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count >= 2, "input tensors length should be >= 2"
    
    input_tensor = input_tensors[0]
    weight_tensor = input_tensors[1]
    
    output_tensors = get_output_tensors(op, tflite_model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    new_tensors.extend(output_tensors)
    output_tensor = output_tensors[0]
    
    # options
    if is_conv2d:
        assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
        op_options = op.BuiltinOptions()
        assert op_options != None, "op has no options"
        conv_options = Conv2DOptions()
        conv_options.Init(op_options.Bytes, op_options.Pos)
    else:
        assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
        op_options = op.BuiltinOptions()
        assert op_options != None, "op has no options"
        conv_options = DepthwiseConv2DOptions()
        conv_options.Init(op_options.Bytes, op_options.Pos)
    
    # conv parameters
    stride_h = conv_options.StrideH()
    stride_w = conv_options.StrideW()
    
    # shape check
    if is_conv2d:
        assert output_tensor.dim_c == weight_tensor.dim_n, "output channels not match"
        assert weight_tensor.dim_c == input_tensor.dim_c, "kernel channels not match"
        assert input_tensor.dim_n == output_tensor.dim_n, "output batches not match"
    else:
        assert input_tensor.dim_c == output_tensor.dim_c, "output channels not match"        
        assert input_tensor.dim_c == weight_tensor.dim_c, "kernel channels not match"
        assert input_tensor.dim_n == output_tensor.dim_n, "output batches not match"
    
    # tensor types
    input_type = input_tensor.data_type
    output_type = output_tensor.data_type
    weight_type = weight_tensor.data_type
    assert input_type == output_type == weight_type, "tensor type not consistent"
    
    bias_idx = np.float64(-1)
    if len(input_tensors) == 3:
        bias_idx = input_tensors[2].tflite_tensor_idx
    
    if is_conv2d:
        conv2d_op = Conv2D(
            input_tensor.tflite_tensor_idx,
            weight_tensor.tflite_tensor_idx,
            output_tensor.tflite_tensor_idx,
            0, 0, stride_h, stride_w, bias_idx
        )
    else:
        conv2d_op = DepthConv2D(
            input_tensor.tflite_tensor_idx,
            weight_tensor.tflite_tensor_idx,
            output_tensor.tflite_tensor_idx,
            0, 0, stride_h, stride_w, bias_idx
        )

    
    ir_model.add_tensors(new_tensors)
    ir_model.add_operator(conv2d_op)
    