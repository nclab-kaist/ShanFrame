from tflite import Operator as TFLiteOperator
from ir.model import Layer


def parse_conv_2d(op: TFLiteOperator) -> Layer:
    raise NotImplementedError(parse_conv_2d)
