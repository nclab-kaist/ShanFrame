import math

import numpy as np

from ..ir import Model, Tensor, OperatorType
from ..ir.operator import Conv2D, DepthConv2D


def fuse_pad(model: Model):
    fused_op_idx = []
    for op_idx, op in model.operators.items():
        if op.op_type != OperatorType.PAD:
            continue
        # op is pad op
        input_idx = op.input_idx_list[0]
        input_tensor = model.tensors[input_idx]
        output_idx = op.output_idx
        output_tensor = model.tensors[output_idx]

        # check if can be fully fused
        can_fully_fuse = True
        for dst_op_idx in output_tensor.dst_op:
            dst_op = model.operators[dst_op_idx]
            if dst_op.op_type != OperatorType.CONV_2D and dst_op.op_type != OperatorType.DEPTH_CONV_2D:
                can_fully_fuse = False
                break
        if not can_fully_fuse:
            continue

        # fuse padding into (depthwise) conv2d
        diff_h = input_tensor.dim_h - output_tensor.dim_h
        assert diff_h % 2 == 0, "height difference not even"
        pad_h = diff_h // 2
        diff_w = input_tensor.dim_w - output_tensor.dim_w
        assert diff_w % 2 == 0, "weight difference not even"
        pad_w = diff_w // 2
        for dst_op_idx in output_tensor.dst_op:
            dst_op = model.operators[dst_op_idx]
            if isinstance(dst_op, DepthConv2D) or isinstance(dst_op, Conv2D):
                dst_op.pad_h = pad_h
                dst_op.pad_w = pad_w
                dst_op.input_idx_list[0] = input_idx
                dst_op.input_idx = input_idx
                input_tensor.dst_op.add(dst_op_idx)
                output_tensor.dst_op.remove(dst_op_idx)
        
        # delete output tensor
        assert len(
            output_tensor.dst_op) == 0, "pad output should have no more usage"
        model.tensors.pop(output_idx)
        # mark op as to delete
        fused_op_idx.append(op_idx)
    
    for op_idx in fused_op_idx:
        model.operators.pop(op_idx)
