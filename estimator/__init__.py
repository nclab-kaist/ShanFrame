from model import Model, Layer

class EstimateResult:
    sram: int
    flash: int

def estimate_model(model: Model) -> EstimateResult:
    raise SystemExit("not implement")

def estimate_layer(layer: Layer) -> EstimateResult:
    raise SystemExit("not implement")
