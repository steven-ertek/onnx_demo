from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx
import os

# 配置路径
model_fp32_path = "models/model.onnx"
model_preprocessed_path = "models/model_preprocessed.onnx"
model_int8_path = "models/model_int8.onnx"

if not os.path.exists(model_fp32_path):
    print(f"Error: Model not found at {model_fp32_path}")
    exit(1)

print(f"Loading FP32 model from: {model_fp32_path}")

# 检查原始模型信息
model = onnx.load(model_fp32_path)
print(f"Original model opset version: {model.opset_import[0].version}")
print(f"Number of nodes: {len(model.graph.node)}")
print(f"Number of initializers: {len(model.graph.initializer)}")

# 预处理：优化模型结构，推断形状信息
print("\nPreprocessing model for quantization...")
quant_pre_process(
    input_model_path=model_fp32_path,
    output_model_path=model_preprocessed_path,
    auto_merge=True,
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=False
)
print(f"Preprocessed model saved to: {model_preprocessed_path}")

print("\nApplying dynamic quantization (INT8)...")

# 动态量化：仅量化特定类型的算子，避免不必要的转换
quantize_dynamic(
    model_input=model_preprocessed_path,
    model_output=model_int8_path,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
    op_types_to_quantize=['Conv', 'MatMul', 'Gemm']  # 只量化卷积和矩阵运算
)

print(f"Quantized model saved to: {model_int8_path}")

# 比较模型大小
fp32_size = os.path.getsize(model_fp32_path) / (1024 * 1024)
preprocessed_size = os.path.getsize(model_preprocessed_path) / (1024 * 1024)
int8_size = os.path.getsize(model_int8_path) / (1024 * 1024)

print(f"\nModel size comparison:")
print(f"  FP32 (original):     {fp32_size:.2f} MB")
print(f"  FP32 (preprocessed): {preprocessed_size:.2f} MB")
print(f"  INT8 (quantized):    {int8_size:.2f} MB")

if int8_size < fp32_size:
    reduction = (1 - int8_size / fp32_size) * 100
    print(f"  Reduction vs original: {reduction:.1f}%")
else:
    increase = (int8_size / fp32_size - 1) * 100
    print(f"  Increase vs original: {increase:.1f}%")
    print("\nNote: Quantized model is larger than original.")
    print("This may happen if the model is very small or already optimized.")
    print("The INT8 model may still run faster due to optimized operations.")



