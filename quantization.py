import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "Output/onnx_model.onnx"
model_int8 = "Output/onnx_model_int8.onnx"

quantized_model = quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)