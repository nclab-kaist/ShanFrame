from enum import Enum, StrEnum
from typing import Callable, Self
from .definition import Operand, Expression
from .operand import Array3DOperand, ElementOperand, OperandType
from ..utils import build_indent


class ExpressionGroup(Expression):
    content: list[Expression]

    def __init__(self, content: list[Expression] | None = None):
        if content is None:
            self.content = list()
        else:
            self.content = content

    def uses_operand(self, operand: Operand) -> bool:
        for expression in self.content:
            if expression.uses_operand(operand):
                return True
        return False

    def to_str(self, indent: int) -> str:
        result = f"{build_indent(indent)}{{\n"
        for expr in self.content:
            result += expr.to_str(indent + 1)
        result += f"{build_indent(indent)}}}\n"
        return result


class ForLoop(ExpressionGroup):
    index: ElementOperand
    start: ElementOperand
    stop: ElementOperand
    step: ElementOperand
    content: list[Expression]

    def __init__(self,
                 index: ElementOperand,
                 start: ElementOperand,
                 stop: ElementOperand,
                 step: ElementOperand,
                 content: list[Expression]
                 ) -> None:
        self.index = index
        self.start = start
        self.stop = stop
        self.step = step
        self.content = content

    def uses_operand(self, operand: Operand) -> bool:
        if self.start == operand or self.stop == operand or self.step == operand:
            return True
        for expression in self.content:
            if expression.uses_operand(operand):
                return True
        return False

    def to_str(self, indent: int) -> str:
        index_str = self.index.to_str()
        start_str = self.start.to_str()
        stop_str = self.stop.to_str()
        step_str = self.step.to_str()
        init_str = f"{self.index.type.to_str()} {index_str} = {start_str}"
        bound_str = f"{index_str} < {stop_str}"
        incr_str = f"{index_str} += {step_str}"

        result = build_indent(indent)
        result += f"for ({init_str}; {bound_str}; {incr_str}) {{ \n"
        for expr in self.content:
            result += expr.to_str(indent + 1)
        result += f"{build_indent(indent)} }}\n"

        return result
    
class IfExpression(ExpressionGroup):
    condition: ElementOperand
    content: list[Expression]
    
    def __init__(self, condition: ElementOperand) -> None:
        self.condition = condition
    
    def uses_operand(self, operand: Operand) -> bool:
        if self.condition == operand:
            return True
        return super().uses_operand(operand)
    
    def to_str(self, indent: int) -> str:
        condition_str = self.condition.to_str()
        result = build_indent(indent)
        result += f"if ({condition_str}) {{ \n"
        for expr in self.content:
            result += expr.to_str(indent + 1)
        result += f"{build_indent(indent)} }}\n"
        return result


def generate_temp_name() -> str:
    temp_count = getattr(generate_temp_name, "counter", 0)
    result_name = "temp" + str(temp_count)
    setattr(generate_temp_name, "counter", temp_count + 1)
    return result_name


class BinaryOperator(StrEnum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    Mod = "%"
    LogicAnd = "&&"
    LogicOr = "||"
    Lt = "<"
    Leq = "<="
    Gt = ">"
    Geq = ">="
    Eq = "=="
    Neq = "!="
    BitAnd = "&"
    BitOr = "|"
    
    def is_arith(self) -> bool:
        arith_op = {
            BinaryOperator.Add,
            BinaryOperator.Sub,
            BinaryOperator.Mul,
            BinaryOperator.Div,
            BinaryOperator.Div,
            BinaryOperator.Mod
        }
        return self in arith_op
    
    def is_logic(self) -> bool:
        logic_op = {
            BinaryOperator.LogicAnd,
            BinaryOperator.LogicOr,
        }
        return self in logic_op
    
    def is_cmp(self) -> bool:
        cmp_op = {
            BinaryOperator.Lt,
            BinaryOperator.Leq,
            BinaryOperator.Gt,
            BinaryOperator.Geq,
            BinaryOperator.Eq,
            BinaryOperator.Neq
        }
        return self in cmp_op
    
    def is_bitwise(self) -> bool:
        bitwise_op = {
            BinaryOperator.BitAnd,
            BinaryOperator.BitOr
        }
        return self in bitwise_op


class BinaryExpression(Expression):
    result: ElementOperand
    operator: BinaryOperator
    left_operand: ElementOperand
    right_operand: ElementOperand

    def __init__(self,
                 operator: BinaryOperator,
                 left_operand: ElementOperand,
                 right_operand: ElementOperand,
                 result_name: str = "") -> None:
        self.operator = operator
        self.left_operand = left_operand
        self.right_operand = right_operand
        self._generate_result(result_name)

    def _generate_result(self, result_name: str) -> None:
        if len(result_name) == 0:
            result_name = generate_temp_name()
        left = self.left_operand
        right = self.right_operand
        if self.operator.is_arith():
            assert left.type == right.type and left.type.bit_len() > 1
            if left.type.is_int():
                self.result = ElementOperand.new_int(left.type.bit_len(), result_name)
            else:
                self.result = ElementOperand.new_float(left.type.bit_len(), result_name)
        elif self.operator.is_logic():
            assert left.type == right.type and left.type.bit_len() == 1
            self.result = ElementOperand.new_int(1, result_name)
        elif self.operator.is_cmp():
            assert left.type == right.type
            self.result = ElementOperand.new_int(1, result_name)
        elif self.operator.is_bitwise():
            assert left.type == right.type and left.type.is_int()
            self.result = ElementOperand.new_int(left.type.bit_len(), result_name)
        raise RuntimeError("Unknown operator type")

    def uses_operand(self, operand: Operand) -> bool:
        return self.left_operand == operand or self.right_operand == operand

    def to_str(self, indent: int) -> str:
        left_str = self.left_operand.to_str()
        right_str = self.right_operand.to_str()
        result_str = self.result.to_str()
        operator_str = str(self.operator)

        return f"{build_indent(indent)}{result_str} = {left_str} {operator_str} {right_str};\n"


class UnaryOperator(StrEnum):
    Minus = "-"
    Reverse = "~"
    Not = "!"
    Same = ""


class UnaryExpression(Expression):
    result: ElementOperand
    operator: UnaryOperator
    operand: ElementOperand

    def __init__(self,
                 operator: UnaryOperator,
                 operand: ElementOperand,
                 result_name: str = "") -> None:
        self.operator = operator
        self.operand = operand
        self._generate_result(result_name)

    def _generate_result(self, result_name: str) -> None:
        if len(result_name) == 0:
            result_name = generate_temp_name()
        match self.operator:
            case UnaryOperator.Reverse:
                assert self.operand.type.is_int()
                self.result = ElementOperand.new_int(self.operand.type.bit_len(), result_name)
            case UnaryOperator.Same | UnaryOperator.Minus:
                self.result = ElementOperand(self.operand.type, self.operand.max_value, self.operand.min_value, result_name)
            case UnaryOperator.Not:
                self.result = ElementOperand.new_int(1, result_name)

    def uses_operand(self, operand: Operand) -> bool:
        return self.operand == operand

    def to_str(self, indent: int) -> str:
        result_str = self.result.to_str()
        operator_str = str(self.operator)
        operand_str = self.operand.to_str()
        return f"{build_indent(indent)}{result_str} = {operator_str}{operand_str};\n"


class BuiltinFunction(StrEnum):
    MAX = "MAX"
    MIN = "MIN"
    # fake function wrapper for array set/copy
    # usage:
    #   memset(array, int offset, int c, int n)
    #   memcpy(dst_array, int dst_offset, src_array, int src_offset, int count)
    # Memset = "memset"
    # Memcpy = "memcpy"
    # fake function for array access, illegal name to avoid collision
    # usage:
    #   3get(1darray, 0, 0, i) === 1darray[i]
    #   3get(2darray, 0, i, j) === 2darray[i][j]
    #   3get(3darray, i, j, k) === 3darray[i][j][k]
    #   3get(weight, ch, x, y) === weight[ch][x][y]
    Arrayget = "3get"
    Arrayset = "3set"


class FunctionExpression(Expression):
    result: Operand
    function: BuiltinFunction
    args: list[Operand]

    def __init__(self, function: BuiltinFunction, args: list[Operand], result_name: str = "") -> None:
        self.function = function
        self.args = args
        self._generate_result(result_name)

    def _generate_result(self, result_name: str) -> None:
        if len(result_name) == 0:
            result_name = generate_temp_name()
        match self.function:
            case BuiltinFunction.MAX | BuiltinFunction.MIN:
                assert len(self.args) == 2
                assert isinstance(self.args[0], ElementOperand) and isinstance(self.args[1], ElementOperand)
                assert self.args[0].type == self.args[1].type
                type = self.args[0].type
                self.result = ElementOperand(type, type.max_value(), type.min_value(), result_name)
            # case BuiltinFunction.Memset:
            #     assert len(self.args) == 4
            #     assert isinstance(self.args[0], Array3DOperand)
            #     assert isinstance(self.args[1], ElementOperand) and self.args[1].type.is_int()
            #     assert isinstance(self.args[2], ElementOperand) and self.args[2].type.is_int()
            #     assert isinstance(self.args[3], ElementOperand) and self.args[3].type.is_int()
            #     self.result = ElementOperand(OperandType.VOID, 0, 0)
            # case BuiltinFunction.Memcpy:
            #     assert len(self.args) == 5
            #     assert isinstance(self.args[0], Array3DOperand)
            #     assert isinstance(self.args[1], ElementOperand) and self.args[1].type.is_int()
            #     assert isinstance(self.args[2], Array3DOperand)
            #     assert isinstance(self.args[3], ElementOperand) and self.args[1].type.is_int()
            #     assert isinstance(self.args[4], ElementOperand) and self.args[1].type.is_int()
            #     self.result = ElementOperand(OperandType.VOID, 0, 0)
            case BuiltinFunction.Arrayget:
                assert len(self.args) == 4
                assert isinstance(self.args[0], Array3DOperand)
                assert isinstance(self.args[1], ElementOperand) and self.args[1].type.is_int()
                assert isinstance(self.args[2], ElementOperand) and self.args[1].type.is_int()
                assert isinstance(self.args[3], ElementOperand) and self.args[1].type.is_int()
                type = self.args[0].type
                self.result = ElementOperand(type, type.max_value(), type.min_value(), result_name)

    def to_str(self, indent: int) -> str:
        result_str = ""
        if not self.result is None:
            result_str = f"{self.result.to_str()} = "
        arg_str = ", ".join([arg.to_str() for arg in self.args])
        return f"{build_indent(indent)}{result_str}{str(self.function)}({arg_str});\n"


class CastExpression(Expression):
    source: ElementOperand
    result: ElementOperand
    
    def __init__(self, 
                 source: ElementOperand, 
                 target_type: OperandType, 
                 result_name: str = ""):
        self.source = source
        self._generate_result(target_type, result_name)
        
    def _generate_result(self, target_type: OperandType, result_name: str) -> None:
        if len(result_name) == 0:
            result_name = generate_temp_name()
        assert target_type.bit_len() > 0 and self.source.type.bit_len() > 0
        self.result = ElementOperand(target_type, target_type.max_value(), target_type.min_value(), result_name)
    
    def to_str(self, indent: int) -> str:
        result_str = self.result.to_str()
        type_str = self.result.type.to_str()
        source_str = self.source.to_str()
        return f"{build_indent(indent)}{result_str} = ({type_str}){source_str};\n"
    