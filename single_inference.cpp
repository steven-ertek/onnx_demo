#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace {
class OnnxRunner {
public:
    explicit OnnxRunner(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_runner"), session_options_(), session_(nullptr) {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_options_.SetIntraOpNumThreads(0);
        session_options_.SetInterOpNumThreads(0);

        const std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc = session_->GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session_->GetOutputNameAllocated(0, allocator);
        input_name_ = std::string(input_name_alloc.get());
        output_name_ = std::string(output_name_alloc.get());

        auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        input_shape_ = input_info.GetShape();
        output_shape_ = output_info.GetShape();

        if (input_shape_.size() != 4) {
            throw std::runtime_error("Model input must be NCHW");
        }
        if (output_shape_.size() != 4) {
            throw std::runtime_error("Model output must be NCHW");
        }

        auto pick_dimension = [](int64_t preferred, int64_t secondary) -> int {
            if (preferred > 0) {
                return static_cast<int>(preferred);
            }
            if (secondary > 0) {
                return static_cast<int>(secondary);
            }
            return -1;
        };

        channels_ = pick_dimension(input_shape_[1], -1);
        if (channels_ <= 0) {
            channels_ = 3; // Default to RGB when model reports dynamic channel dimension
            std::cout << "Input channels reported as dynamic; falling back to 3 (RGB)." << std::endl;
        }

        height_ = pick_dimension(input_shape_[2], output_shape_[2]);
        width_ = pick_dimension(input_shape_[3], output_shape_[3]);

        if (channels_ <= 0 || height_ <= 0 || width_ <= 0) {
            throw std::runtime_error("Failed to determine concrete input dimensions");
        }

        input_dims_ = {1, channels_, height_, width_};
        std::cout << "Using ONNX Runtime (CPU)" << std::endl;
    }

    cv::Mat infer(const std::vector<float>& input_data) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_dims_.data(),
            input_dims_.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        auto& output_tensor = output_tensors.front();
        auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();
        if (output_shape.size() != 4) {
            throw std::runtime_error("Unexpected output shape");
        }

        const int out_h = static_cast<int>(output_shape[2]);
        const int out_w = static_cast<int>(output_shape[3]);
        const float* data = output_tensor.GetTensorData<float>();

        cv::Mat prob(out_h, out_w, CV_32F);
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                // 模型输出已包含 sigmoid，直接使用
                prob.at<float>(h, w) = data[h * out_w + w];
            }
        }
        return prob;
    }

    int height() const { return height_; }
    int width() const { return width_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    std::array<int64_t, 4> input_dims_{};
    int channels_{0};
    int height_{0};
    int width_{0};
};

cv::Mat preprocess(const cv::Mat& image, int target_width, int target_height) {
    // 先转换为 RGB，再 resize（与 PyTorch 版本一致）
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(target_width, target_height));
    // 归一化到 [0, 1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    return resized;
}

std::vector<float> to_chw(const cv::Mat& rgb) {
    const int channels = rgb.channels();
    const int height = rgb.rows;
    const int width = rgb.cols;
    std::vector<float> buffer(channels * height * width);
    size_t idx = 0;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            const float* row_ptr = rgb.ptr<float>(h);
            for (int w = 0; w < width; ++w) {
                buffer[idx++] = row_ptr[w * channels + c];
            }
        }
    }
    return buffer;
}

void save_outputs(const cv::Mat& original, const cv::Mat& prob, const std::string& prefix) {
    cv::Mat prob_u8;
    prob.convertTo(prob_u8, CV_8U, 255.0);

    cv::Mat binary;
    cv::threshold(prob, binary, 0.5, 1.0, cv::THRESH_BINARY);
    cv::Mat binary_u8;
    binary.convertTo(binary_u8, CV_8U, 255.0);

    cv::Mat original_resized;
    cv::resize(original, original_resized, prob.size());

    cv::Mat heatmap;
    cv::applyColorMap(prob_u8, heatmap, cv::COLORMAP_JET);
    cv::Mat overlay;
    cv::addWeighted(original_resized, 0.7, heatmap, 0.3, 0.0, overlay);

    cv::Mat combined;
    cv::hconcat(original_resized, overlay, combined);
    cv::imwrite(prefix + "_result.png", combined);
}
} // namespace

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>        Path to ONNX model (default: models/best.onnx)\n";
    std::cout << "  --image <path>        Path to input image (required)\n";
    std::cout << "  --output <prefix>     Output file prefix (default: onnxruntime_cpu_inference)\n";
    std::cout << "  --save <0|1>          Save prediction images (default: 1, 0=no save)\n";
    std::cout << "  --help                Show this help message\n";
}

int main(int argc, char* argv[]) {
    // 默认值
    std::string model_path = "models/best.onnx";
    std::string image_path;
    std::string output_prefix = "onnxruntime_cpu_inference";
    bool enable_save = true;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            enable_save = (std::stoi(argv[++i]) != 0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        // 检查必需参数
        if (image_path.empty()) {
            std::cerr << "Error: --image parameter is required\n\n";
            print_usage(argv[0]);
            return 1;
        }

        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model not found: " << model_path << std::endl;
            std::cout << "Press Enter to exit...";
            std::cin.get();
            return 1;
        }
        if (!std::filesystem::exists(image_path)) {
            std::cerr << "Image not found: " << image_path << std::endl;
            std::cout << "Press Enter to exit...";
            std::cin.get();
            return 1;
        }

        OnnxRunner runner(model_path);
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            std::cout << "Press Enter to exit...";
            std::cin.get();
            return 1;
        }

        cv::Mat rgb = preprocess(image, runner.width(), runner.height());
        auto chw = to_chw(rgb);

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat prob = runner.infer(chw);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[ONNX Runtime - CPU] Inference time: " << duration << " ms" << std::endl;

        if (enable_save) {
            save_outputs(image, prob, output_prefix);
            std::cout << "Result saved: " << output_prefix << "_result.png" << std::endl;
        } else {
            std::cout << "Skipping save (--save 0)" << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return 1;
    }
    std::cout << "Press Enter to exit...";
    std::cin.get();
    return 0;
}
