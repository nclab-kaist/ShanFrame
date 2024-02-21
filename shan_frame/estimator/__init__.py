from ir.model import Model, Layer


class EstimateResult:
    sram: int
    flash: int
    
class Estimator:
    output_dir: str
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def estimate_model(self, model: Model) -> EstimateResult:
        raise SystemExit("not implement")


    def estimate_layer(self, layer: Layer) -> EstimateResult:
        raise SystemExit("not implement")
