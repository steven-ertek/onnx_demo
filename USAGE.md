# 使用说明

所有推理程序现在都支持命令行参数输入，无需修改代码即可灵活配置。

## 单图推理

### ONNX Runtime 版本
```bash
# 基本用法（必须指定图片）
.\build\x64\Release\single_inference.exe --image path/to/image.png

# 完整参数
.\build\x64\Release\single_inference.exe ^
  --model models/best.onnx ^
  --image D:/ertek_data/scratch_data/images/test.png ^
  --output my_result

# 查看帮助
.\build\x64\Release\single_inference.exe --help
```

**参数说明：**
- `--model <path>`: ONNX模型路径（默认：models/best.onnx）
- `--image <path>`: 输入图片路径（**必需**）
- `--output <prefix>`: 输出文件前缀（默认：onnxruntime_cpu_inference）
- `--help, -h`: 显示帮助信息

### OpenVINO 版本
```bash
# 基本用法
.\build\x64\Release\single_inference_openvino.exe --image path/to/image.png

# 完整参数
.\build\x64\Release\single_inference_openvino.exe ^
  --model models/best.onnx ^
  --image D:/ertek_data/scratch_data/images/test.png ^
  --output openvino_result
```

**参数说明：**同上（输出前缀默认：openvino_cpu_inference）

---

## 批量推理

### ONNX Runtime 版本
```bash
# 基本用法（使用默认配置）
.\build\x64\Release\batch_inference.exe

# 自定义参数
.\build\x64\Release\batch_inference.exe ^
  --model models/micro_unet.onnx ^
  --images D:/ertek_data/scratch_data/images ^
  --list test.txt ^
  --output results_custom ^
  --batch 16 ^
  --timing timing_report.txt

# 查看帮助
.\build\x64\Release\batch_inference.exe --help
```

**参数说明：**
- `--model <path>`: ONNX模型路径（默认：models/model.onnx）
- `--images <path>`: 图片根目录（默认：D:/ertek_data/scratch_data/images）
- `--list <path>`: 包含图片文件名的文本文件（默认：test.txt）
- `--output <path>`: 结果输出目录（默认：batch_results_onnx）
- `--batch <size>`: 批处理大小（默认：8）
- `--timing <path>`: 时间统计报告文件（默认：onnx_batch_timing.txt）
- `--help, -h`: 显示帮助信息

### OpenVINO 版本
```bash
# 基本用法
.\build\x64\Release\batch_inference_openvino.exe

# 自定义参数
.\build\x64\Release\batch_inference_openvino.exe ^
  --model models/micro_unet.onnx ^
  --images D:/ertek_data/scratch_data/images ^
  --list test.txt ^
  --output results_openvino ^
  --batch 32 ^
  --timing openvino_timing.txt
```

**参数说明：**同上（默认输出目录：batch_results_openvino，默认报告：openvino_batch_timing.txt）

---

## 使用示例

### 场景1：测试不同模型
```bash
# 测试 micro_unet 模型
.\build\x64\Release\batch_inference.exe --model models/micro_unet.onnx --output results_micro

# 测试 standard unet 模型
.\build\x64\Release\batch_inference.exe --model models/unet.onnx --output results_unet
```

### 场景2：比较不同 batch size 性能
```bash
# Batch size = 1
.\build\x64\Release\batch_inference.exe --batch 1 --timing timing_batch1.txt

# Batch size = 8
.\build\x64\Release\batch_inference.exe --batch 8 --timing timing_batch8.txt

# Batch size = 32
.\build\x64\Release\batch_inference.exe --batch 32 --timing timing_batch32.txt
```

### 场景3：处理不同数据集
```bash
# 训练集
.\build\x64\Release\batch_inference.exe ^
  --list train_list.txt ^
  --output results_train

# 测试集
.\build\x64\Release\batch_inference.exe ^
  --list test_list.txt ^
  --output results_test
```

### 场景4：快速单图测试
```bash
# 使用默认模型测试单张图片
.\build\x64\Release\single_inference.exe --image test_image.png

# 使用自定义模型
.\build\x64\Release\single_inference.exe ^
  --model models/micro_unet.onnx ^
  --image test_image.png ^
  --output micro_result
```

---

## 注意事项

1. **路径格式**：Windows 下可以使用 `/` 或 `\\` 作为路径分隔符
2. **相对路径**：相对于程序运行目录（通常是 build/x64/Release/）
3. **必需参数**：单图推理必须指定 `--image` 参数
4. **图片列表文件**：批量推理的 list 文件每行一个相对于 images 目录的图片文件名
5. **输出目录**：如果不存在会自动创建

## 重新编译
修改代码后需要重新编译：
```bash
cd build
cmake --build . --config Release
```
