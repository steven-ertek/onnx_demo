# ONNX demo on Intel CPU with Windows 11

## Preparation
<img src="./asserts/vcpkg.svg" width="10px" height="auto">  vcpkg   https://github.com/microsoft/vcpkg

<img src="./asserts/onnxruntime.svg" width="15px" height="auto">    ONNX Runtime    https://onnxruntime.ai/docs/install/

<img src="./asserts/openvino.svg" width="80px" height="auto">   OpenVINO    https://github.com/openvinotoolkit/openvino


## Quick Start
```
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=path/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

## Directory
```
onnx_demo
    |-----asserts
    |
    |-----utils
    |       |-----xxx.py
    |       |-----xxx.py
    |
    |-----CMakeLists.txt
    |-----xxx.cpp
    |-----xxx.cpp
```