def signed_bit_length(value: int) -> int:
    if value >= 0:
        return value.bit_length()
    else:
        return value.bit_length() + 1


def build_indent(indent: int) -> str:
    return "    " * indent
