import onnx
import onnxruntime as ort
import numpy as np
import cv2
import os

# 检查模型文件大小
print("=" * 60)
print("ONNX Model File Sizes")
print("=" * 60)

model_files = ["models/best.onnx", "models/model.onnx", "models/best_fixed.onnx"]
for model_file in model_files:
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        size_kb = os.path.getsize(model_file) / 1024
        print(f"{os.path.basename(model_file):20s} {size_mb:>8.2f} MB ({size_kb:>10.2f} KB)")
    else:
        print(f"{model_file}: Not found")

# 加载 ONNX 模型
model_path = "models/best.onnx"
model = onnx.load(model_path)

print("\n" + "=" * 60)
print("ONNX Model Analysis")
print("=" * 60)

# 检查输入输出
print("\n[Input Info]")
for input_tensor in model.graph.input:
    print(f"  Name: {input_tensor.name}")
    print(f"  Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in input_tensor.type.tensor_type.shape.dim]}")
    print(f"  Type: {input_tensor.type.tensor_type.elem_type}")

print("\n[Output Info]")
for output_tensor in model.graph.output:
    print(f"  Name: {output_tensor.name}")
    print(f"  Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in output_tensor.type.tensor_type.shape.dim]}")
    print(f"  Type: {output_tensor.type.tensor_type.elem_type}")

# 检查最后几层的算子类型
print("\n[Last 5 Nodes]")
for node in model.graph.node[-5:]:
    print(f"  {node.op_type}: {node.input} -> {node.output}")

# 检查是否有 Sigmoid/Softmax 在最后
has_sigmoid = any(node.op_type == "Sigmoid" for node in model.graph.node[-3:])
has_softmax = any(node.op_type == "Softmax" for node in model.graph.node[-3:])
print(f"\n[Activation Check]")
print(f"  Has Sigmoid in last 3 nodes: {has_sigmoid}")
print(f"  Has Softmax in last 3 nodes: {has_softmax}")

# 测试推理
print("\n" + "=" * 60)
print("Test Inference")
print("=" * 60)

image_path = "D:/ertek_data/scratch_data/images/Temp0027F_0_0_0_832.png"
image = cv2.imread(image_path)
if image is None:
    print(f"Cannot load image: {image_path}")
    exit(1)

# 预处理（与 C++ 一致）
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (256, 256))  # 假设模型输入是 256x256
normalized = resized.astype(np.float32) / 255.0

# 转换为 CHW
chw = np.transpose(normalized, (2, 0, 1))
input_tensor = np.expand_dims(chw, axis=0)  # 添加 batch 维度

print(f"\n[Preprocessed Input]")
print(f"  Shape: {input_tensor.shape}")
print(f"  Dtype: {input_tensor.dtype}")
print(f"  Min: {input_tensor.min():.6f}")
print(f"  Max: {input_tensor.max():.6f}")
print(f"  Mean: {input_tensor.mean():.6f}")

# 推理
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run([output_name], {input_name: input_tensor})[0]

print(f"\n[Raw Output]")
print(f"  Shape: {output.shape}")
print(f"  Dtype: {output.dtype}")
print(f"  Min: {output.min():.6f}")
print(f"  Max: {output.max():.6f}")
print(f"  Mean: {output.mean():.6f}")
print(f"  Sample values (first 10): {output.flatten()[:10]}")

# 如果输出值在 [0, 1] 范围内，说明模型已经包含 sigmoid
if output.min() >= 0 and output.max() <= 1:
    print("\n⚠️  Output is already in [0, 1] range - model likely has Sigmoid!")
    print("    C++ code should NOT apply sigmoid again!")
else:
    print("\n✓  Output is raw logits - needs Sigmoid activation")
    sigmoid_output = 1.0 / (1.0 + np.exp(-output))
    print(f"  After Sigmoid - Min: {sigmoid_output.min():.6f}, Max: {sigmoid_output.max():.6f}")

print("\n" + "=" * 60)
