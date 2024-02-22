from shan_frame.ir.model import Model, Layer


class Estimator:
    output_dir: str

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def estimate_model(self, model: Model) -> tuple[int, int]:
        raise SystemExit("not implement")

    def estimate_layer(self, layer: Layer) -> tuple[int, int]:
        raise SystemExit("not implement")
