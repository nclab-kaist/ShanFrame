from ..ir import Operator, Model
from ..ir.operator import *


def gen_copy_int8(src: str, dst: str, size: str) -> list[str]:
    # XXX: rely on memcpy, may not be most efficient
    return [f"memcpy({dst}, {src}, {size})"]

def concat_line(prev: str, next: str, indent: int) -> str:
    indent_str = "    "
    return f"{prev}{indent_str * indent}{next}\n"

def indent_lines(input: str, indent: int) -> str:
    indent_str = "    "
    lines = [indent_str * indent + line.strip() + "\n" for line in input.split("\n")]
    return "".join(lines)