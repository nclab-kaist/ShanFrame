from enum import Enum, StrEnum
from typing import Callable, Self
from .definition import Operand, Expression
from .operand import ElementOperand, OperandType
from ..utils import build_indent
from .gen_result import *


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


class ConstantLoop(ExpressionGroup):
    index: ElementOperand
    step: int
    loop_range: range
    unroll_level: int
    content: list[Expression]

    def __init__(self,
                 index: ElementOperand,
                 range: range,
                 base_content: list[Expression] | None = None,
                 step: int = 1,
                 unroll_level: int = 1
                 ) -> None:
        assert index.type.is_int
        self.index = index
        self.step = step
        self.loop_range = range
        self.unroll_level = unroll_level
        if base_content is None:
            self.base_content = list()
        else:
            self.base_content = base_content

    def uses_operand(self, operand: Operand) -> bool:
        for expression in self.base_content:
            if expression.uses_operand(operand):
                return True
        return False

    def to_str(self, indent: int) -> str:
        index_str = self.index.to_str()
        init_str = f"{self.index.type.to_str()} {index_str} = {self.loop_range.start}"
        bound_str = f"{index_str} < {self.loop_range.stop}"
        incr_str = f"{index_str} ++"

        result = build_indent(indent)
        result += f"for ({init_str}; {bound_str}; {incr_str}) {{ \n"
        for expr in self.content:
            result += expr.to_str(indent + 1)
        result += f"{build_indent(indent)} }}\n"

        raise NotImplementedError("FiniteLoop.to_str")


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
        # TODO: generate the result of an expression,
        raise NotImplementedError("UnaryExpression._generate_result")

    def uses_operand(self, operand: Operand) -> bool:
        return self.operand == operand

    def to_str(self, indent: int) -> str:
        result_str = self.result.to_str()
        operator_str = str(self.operator)
        operand_str = self.operand.to_str()
        return f"{build_indent(indent)}{result_str} = {operator_str}{operand_str};\n"


class BuiltinFunction(Enum):
    MAX = "MAX"
    MIN = "MIN"
    Memset = "memset"
    Memcpy = "memcpy"
    # fake function for array access, illegal name to avoid collision
    # usage:
    #   3get(3darray, channel, x, y) === 3darray[channel][x][y]
    Arrayget = "3get"
    Arrayset = "3set"


class FunctionExpression(Expression):
    result: Operand | None
    function: BuiltinFunction
    args: list[Operand]

    def __init__(self, function: BuiltinFunction, args: list[Operand]) -> None:
        self.function = function
        self.args = args
        self._generate_result()

    def _generate_result(self) -> None:
        raise NotImplementedError("FunctionExpression._generate_result")

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
        # TODO: generate the result of an expression,
        raise NotImplementedError("CastExpression._generate_result")
    
    def to_str(self, indent: int) -> str:
        result_str = self.result.to_str()
        type_str = self.result.type.to_str()
        source_str = self.source.to_str()
        return f"{build_indent(indent)}{result_str} = ({type_str}){source_str};\n"
    