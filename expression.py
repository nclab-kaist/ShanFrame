from enum import Enum
from abc import ABCMeta, abstractclassmethod
from operand import Operand, ElementOperand


class Expression(ABCMeta):
    result: ElementOperand

    @abstractclassmethod
    def uses_operand(self, operand: Operand) -> bool:
        pass

    @abstractclassmethod
    def to_str(self, indent: int) -> str:
        pass


class ExpressionGroup(Expression):
    result: ElementOperand
    content: list[Expression]

    def __init__(self, content: list[Expression] = None):
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
        # TODO: implement to_str for ExpressionGroup
        raise SystemExit("not implemented")


class FiniteLoop(ExpressionGroup):
    result: ElementOperand
    index: ElementOperand
    step: int
    range: range
    unroll_level: int
    content: list[Expression]

    def __init__(self,
                 indent: int,
                 range: range,
                 base_content: list[Expression] = None,
                 step: int = 1,
                 unroll_level: int = 1
                 ) -> None:
        self.result = None
        self.index = ElementOperand.new_int(32)
        self.step = step
        self.range = range
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
        raise SystemExit("not implemented")


class BinaryOperator(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    GET = 5
    SET = 6
    SADD8 = 7
    SADD16 = 8


class BinaryExpression(Expression):
    result: ElementOperand
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
        pass

    def uses_operand(self, operand: Operand) -> bool:
        return self.left_operand == operand or self.right_operand == operand

    def to_str(self, indent: int) -> str:
        # TODO: implement to_str for BinaryExpression
        raise SystemExit("not implemented")


class UnaryOperator(Enum):
    TypeCast = 1
    LessThan = 2
    LessOrEqualTo = 3
    GreaterThan = 4
    GreaterOrEqualTo = 5
    EqualTo = 6
    NotEqualTo = 7


class UnaryExpression(Expression):
    result: ElementOperand
    operator: UnaryOperator
    operand: ElementOperand

    def __init__(self,
                 operator: UnaryOperator,
                 operand: ElementOperand) -> None:
        self.operator = operator
        self.operand = operand
        self._generate_result()

    def _generate_result(self) -> None:
        # TODO: generate the result of this expression
        pass

    def uses_operand(self, operand: Operand) -> bool:
        return self.operand == operand

    def to_str(self, indent: int) -> str:
        # TODO: implement to_str for UnaryExpression
        raise SystemExit("not implemented")
