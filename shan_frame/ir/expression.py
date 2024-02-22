from enum import Enum
from .definition import Operand, Expression
from .operand import ElementOperand


class ExpressionGroup(Expression):
    result: ElementOperand | None
    content: list[Expression]

    def __init__(self, content: list[Expression] | None = None):
        self.result = None
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
        raise NotImplementedError("ExpressionGroup.to_str")


class ConstantLoop(ExpressionGroup):
    result: ElementOperand | None
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
        self.result = None
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
        # TODO: implement to_str for FiniteExpression
        raise NotImplementedError("FiniteLoop.to_str")


class BinaryOperator(Enum):
    Add = 1
    Sub = 2
    Mul = 3
    Div = 4
    Mod = 5
    Get = 6
    Set = 7
    And = 8
    Or = 9
    Sadd8 = 10
    Sadd16 = 11


class BinaryExpression(Expression):
    result: ElementOperand | None
    operator: BinaryOperator
    left_operand: Operand
    right_operand: ElementOperand

    def __init__(self,
                 operator: BinaryOperator,
                 left_operand: Operand,
                 right_operand: ElementOperand) -> None:
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
        # TODO: implement to_str for BinaryExpression
        raise NotImplementedError("BinaryExpression.to_str")


class UnaryOperator(Enum):
    TypeCast = 1
    LessThan = 2
    LessOrEqualTo = 3
    GreaterThan = 4
    GreaterOrEqualTo = 5
    EqualTo = 6
    NotEqualTo = 7
    Minus = 8
    Reverse = 9


class UnaryExpression(Expression):
    result: ElementOperand | None
    operator: UnaryOperator
    operand: ElementOperand

    def __init__(self,
                 operator: UnaryOperator,
                 operand: ElementOperand) -> None:
        self.operator = operator
        self.operand = operand
        self._generate_result()

    def _generate_result(self) -> None:
        raise NotImplementedError("UnaryExpression._gen_result")

    def uses_operand(self, operand: Operand) -> bool:
        return self.operand == operand

    def to_str(self, indent: int) -> str:
        raise NotImplementedError("UnaryExpression.to_str")


class BuiltinFunction(Enum):
    MAX = 1
    MIN = 2
    Memset = 3
    Memcpy = 4


class FunctionExpression(Expression):
    result: ElementOperand | None
    function: BuiltinFunction
    args: list[ElementOperand]

    def __init__(self, function: BuiltinFunction, args: list[ElementOperand]) -> None:
        self.function = function
        self.args = args
        self._generate_result()

    def _generate_result(self) -> None:
        raise NotImplementedError("FunctionExpression._generate_result")
