from enum import Enum, StrEnum
from typing import Self
from .definition import Operand, Expression
from .operand import ElementOperand, OperandType
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
    And = "&&"
    Or = "||"
    BitAnd = "&"
    BitOr = "|"


class BinaryExpression(Expression):
    result: ElementOperand
    operator: BinaryOperator
    left_operand: Operand
    right_operand: ElementOperand
    _result_count: int

    def __init__(self,
                 operator: BinaryOperator,
                 left_operand: Operand,
                 right_operand: ElementOperand,
                 result_name: str = "") -> None:
        self.operator = operator
        self.left_operand = left_operand
        self.right_operand = right_operand
        self._generate_result(result_name)

    def _generate_result(self, result_name: str) -> None:
        if len(result_name) == 0:
            result_name = generate_temp_name()
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
    