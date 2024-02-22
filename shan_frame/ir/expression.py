from enum import Enum, StrEnum
from typing import Self
from .definition import Operand, Expression
from .operand import ElementOperand
from shan_frame.utils import build_indent


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
                 indent: int,
                 range: range,
                 base_content: list[Expression] | None = None,
                 step: int = 1,
                 unroll_level: int = 1
                 ) -> None:
        self.index = ElementOperand.new_int(32)
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


class BinaryOperator(StrEnum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    Mod = "%"
    And = "&&"
    Or = "||"
    BitAnd = "&"
    BitOr = "|"


class BinaryExpression(Expression):
    result: Operand
    operator: BinaryOperator
    left_operand: Operand
    right_operand: Operand

    def __init__(self,
                 operator: BinaryOperator,
                 left_operand: Operand,
                 right_operand: Operand) -> None:
        self.operator = operator
        self.left_operand = left_operand
        self.right_operand = right_operand
        self._generate_result()

    def _generate_result(self) -> None:
        # TODO: generate the result of an expression,
        raise NotImplementedError("BinaryExpression._generate_result")

    def uses_operand(self, operand: Operand) -> bool:
        return self.left_operand == operand or self.right_operand == operand

    def to_str(self, indent: int) -> str:
        left_str = self.left_operand.to_str()
        right_str = self.right_operand.to_str()
        result_str = self.result.to_str()
        operator_str = str(self.operator)

        return f"{build_indent(indent)}{result_str} = {left_str} {operator_str} {right_str};\n"


class UnaryOperator(StrEnum):
    TypeCast = "cast"
    LessThan = "<"
    LessOrEqualTo = "<="
    GreaterThan = ">"
    GreaterOrEqualTo = ">="
    EqualTo = "=="
    NotEqualTo = "!="
    Minus = "-"
    Reverse = "~"
    Same = ""


class UnaryExpression(Expression):
    result: Operand
    operator: UnaryOperator
    operand: Operand

    def __init__(self,
                 operator: UnaryOperator,
                 operand: Operand) -> None:
        self.operator = operator
        self.operand = operand
        self._generate_result()

    def _generate_result(self) -> None:
        raise NotImplementedError("UnaryExpression._gen_result")

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
