
from tflite import Model as TFliteModel
from tflite import Operator as TFliteOP
from tflite import SubGraph, BuiltinOperator
from ..ir import Model as IRModel

from .parse_conv2d import parse_conv2d
from .parse_avgpool import parse_avgpool2d
from .parse_pad import parse_pad
from .parse_add import parse_add
from .parse_reshape import parse_reshape
from .fuse_pad import fuse_pad

class ModelParser:
    model_path: str
    tflite_model: TFliteModel

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        buf = open(model_path, "rb").read()
        self.tflite_model = TFliteModel.GetRootAs(buf)

    def parse_model(self) -> IRModel:
        subgraph = self.tflite_model.Subgraphs(0)
        if subgraph is None:
            raise RuntimeError("subgraph 0 not exist")
        model: IRModel = IRModel()
        operators_len: int = subgraph.OperatorsLength()
        skip_next_ops = 0
        for i in range(operators_len):
            if skip_next_ops > 0:
                skip_next_ops -= 1
                continue
            op: TFliteOP | None = subgraph.Operators(i)
            if op is None:
                continue
            if i + 2 < operators_len - 2:
                next_op = subgraph.Operators(i + 1)
                if next_op is None:
                    raise RuntimeError("next_op is none")
                next_next_op = subgraph.Operators(i + 2)
                if next_next_op is None:
                    raise RuntimeError("next_next_op is none")
                three_op_sequence = [op, next_op, next_next_op]
                if self._checkIfRequireSElementmult(three_op_sequence):
                    # SE block detected
                    skip_next_ops = 2
                    raise NotImplementedError("parse se block")

            self._handleOperator(op, model)
        fuse_pad(model)
        return model

    def _handleOperator(self, op: TFliteOP, model: IRModel):
        op_code = self._getOpCodeStr(op)
        match op_code:
            case "PAD":
                parse_pad(op, self.tflite_model, model)
            case "CONV_2D" | "DEPTHWISE_CONV_2D":
                parse_conv2d(op, self.tflite_model, model)
            case "AVERAGE_POOL_2D":
                parse_avgpool2d(op, self.tflite_model, model)
            case "ADD":
                parse_add(op, self.tflite_model, model)
            case "RESHAPE":
                parse_reshape(op, self.tflite_model, model)
            case _:
                raise NotImplementedError(f"Unsupported op: {op_code}")

    def _getOpCodeStr(self, op: TFliteOP) -> str:
        op_code_list_idx = op.OpcodeIndex()
        op_codes = self.tflite_model.OperatorCodes(op_code_list_idx)
        if op_codes is None:
            raise RuntimeError("mdoel op codes is none")
        op_code_id = op_codes.DeprecatedBuiltinCode()

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

    def _checkIfRequireSElementmult(self, three_op_sequence: list[TFliteOP]) -> bool:
        if (
            self._getOpCodeStr(three_op_sequence[0]) == "ADD"
            and self._getOpCodeStr(three_op_sequence[1]) == "MUL"
            and self._getOpCodeStr(three_op_sequence[2]) == "MUL"
        ):
            return True
        return False
