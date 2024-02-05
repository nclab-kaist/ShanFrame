from tflite import Model as TFliteModel
from tflite import SubGraph

model_path: str = "/home/locky/temp/ad_small_int8.tflite"
model: TFliteModel = TFliteModel.GetRootAs(model_path)
subgraph: SubGraph = model.Subgraphs(0) # type: ignore
