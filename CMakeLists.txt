cmake_minimum_required(VERSION 3.15)
project(onnx_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 OpenVINO 路径
set(OpenVINO_DIR "D:/Intel/openvino_2025/runtime/cmake")

# 查找 vcpkg 安装的包，仅选择推理需要的模块，避免额外依赖 ONNX
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(onnxruntime REQUIRED)
find_package(OpenVINO REQUIRED)

# ONNX Runtime 版本
add_executable(single_inference single_inference.cpp)
target_link_libraries(single_inference 
    PRIVATE 
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    onnxruntime::onnxruntime
)
target_include_directories(single_inference 
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

# OpenVINO 版本
add_executable(single_inference_openvino single_inference_openvino.cpp)
target_link_libraries(single_inference_openvino 
    PRIVATE 
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    openvino::runtime
)
target_include_directories(single_inference_openvino 
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

# 复制 OpenVINO DLL 到输出目录
add_custom_command(TARGET single_inference_openvino POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "D:/Intel/openvino_2025/runtime/bin/intel64/$<CONFIG>"
    "$<TARGET_FILE_DIR:single_inference_openvino>"
    COMMENT "Copying OpenVINO DLLs to output directory"
)

# 批量推理 - ONNX Runtime
add_executable(batch_inference batch_inference.cpp)
target_link_libraries(batch_inference 
    PRIVATE 
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    onnxruntime::onnxruntime
)
target_include_directories(batch_inference 
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

# 批量推理 - OpenVINO
add_executable(batch_inference_openvino batch_inference_openvino.cpp)
target_link_libraries(batch_inference_openvino 
    PRIVATE 
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    openvino::runtime
)
target_include_directories(batch_inference_openvino 
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

# 复制 OpenVINO DLL 到批量推理输出目录
add_custom_command(TARGET batch_inference_openvino POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "D:/Intel/openvino_2025/runtime/bin/intel64/$<CONFIG>"
    "$<TARGET_FILE_DIR:batch_inference_openvino>"
    COMMENT "Copying OpenVINO DLLs to batch inference output directory"
)

# 设置输出目录
set_target_properties(single_inference single_inference_openvino 
                      batch_inference batch_inference_openvino PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
