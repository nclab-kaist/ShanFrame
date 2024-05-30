from .utils import indent_lines
from .output_code import OutputCode, KernelFunc


def generate_inference_content(ret_addr: str, output_code: OutputCode) -> str:
    content = ""
    indent = 1
    for idx, kernel in output_code.kernels.items():
        content += indent_lines(f"{kernel.call};", indent)
    content += indent_lines(f"return {ret_addr};", indent)
    return content
