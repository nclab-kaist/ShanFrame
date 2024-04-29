from ir import Model


class CodeGenerator:
    output_dir: str
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def generate(self, model: Model) -> None:
        raise NotImplementedError("CodeGenerator.generate()")