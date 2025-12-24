#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class OnnxRunner {
public:
    explicit OnnxRunner(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_batch"), session_options_(), session_(nullptr) {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_options_.SetIntraOpNumThreads(0);
        session_options_.SetInterOpNumThreads(0);
        const std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = std::string(session_->GetInputNameAllocated(0, allocator).get());
        output_name_ = std::string(session_->GetOutputNameAllocated(0, allocator).get());

        auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        input_shape_ = input_info.GetShape();
        output_shape_ = output_info.GetShape();

        if (input_shape_.size() != 4 || output_shape_.size() != 4) {
            throw std::runtime_error("Model must use NCHW tensors");
        }

        channels_ = static_cast<int>(input_shape_[1] > 0 ? input_shape_[1] : output_shape_[1]);
        height_ = static_cast<int>(input_shape_[2] > 0 ? input_shape_[2] : output_shape_[2]);
        width_ = static_cast<int>(input_shape_[3] > 0 ? input_shape_[3] : output_shape_[3]);
        if (channels_ <= 0 || height_ <= 0 || width_ <= 0) {
            throw std::runtime_error("Invalid model input dimensions");
        }

        input_dims_ = {1, channels_, height_, width_};
    }

    cv::Mat infer(const std::vector<float>& input_data) {
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_dims_.data(),
            input_dims_.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        auto& output_tensor = outputs.front();
        auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
        auto shape = output_info.GetShape();
        if (shape.size() != 4) {
            throw std::runtime_error("Unexpected output shape");
        }

        const int out_h = static_cast<int>(shape[2]);
        const int out_w = static_cast<int>(shape[3]);
        const float* raw = output_tensor.GetTensorData<float>();

        cv::Mat prob(out_h, out_w, CV_32F);
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                float value = raw[h * out_w + w];
                prob.at<float>(h, w) = 1.0f / (1.0f + std::exp(-value));
            }
        }
        return prob;
    }

    int width() const { return width_; }
    int height() const { return height_; }

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

cv::Mat preprocess(const cv::Mat& image, int target_w, int target_h) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_w, target_h));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
}

std::vector<float> to_chw(const cv::Mat& rgb) {
    const int channels = rgb.channels();
    const int height = rgb.rows;
    const int width = rgb.cols;
    std::vector<float> buffer(channels * height * width);
    size_t idx = 0;
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            const float* row = rgb.ptr<float>(h);
            for (int w = 0; w < width; ++w) {
                buffer[idx++] = row[w * channels + c];
            }
        }
    }
    return buffer;
}

void save_results(const cv::Mat& original, const cv::Mat& prob, const std::filesystem::path& out_dir, const std::string& stem) {
    cv::Mat prob_u8;
    prob.convertTo(prob_u8, CV_8U, 255.0);
    cv::Mat binary;
    cv::threshold(prob, binary, 0.5, 1.0, cv::THRESH_BINARY);
    cv::Mat binary_u8;
    binary.convertTo(binary_u8, CV_8U, 255.0);

    cv::Mat original_resized;
    cv::resize(original, original_resized, prob.size());

    cv::Mat binary_color;
    cv::cvtColor(binary_u8, binary_color, cv::COLOR_GRAY2BGR);
    cv::Mat combined;
    cv::hconcat(original_resized, binary_color, combined);

    cv::imwrite((out_dir / (stem + "_binary.png")).string(), binary_u8);
    cv::imwrite((out_dir / (stem + "_combined.png")).string(), combined);
}
} // namespace

int main() {
    try {
        const std::filesystem::path model_path = "D:/ertek_codebase/onnx_demo/models/model.onnx";
        const std::filesystem::path image_root = "D:/ertek_data/scratch_data/images";
        const std::filesystem::path list_file = "D:/ertek_codebase/onnx_demo/test.txt";
        const std::filesystem::path output_dir = "D:/ertek_codebase/onnx_demo/batch_results_onnx";
        const std::filesystem::path timing_report = "D:/ertek_codebase/onnx_demo/onnx_batch_timing.txt";
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model not found: " << model_path << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(list_file)) {
            std::cerr << "Test list not found: " << list_file << std::endl;
            return 1;
        }

        std::filesystem::create_directories(output_dir);
        OnnxRunner runner(model_path.string());

        std::ifstream list_stream(list_file);
        std::vector<std::string> image_names;
        std::string line;
        while (std::getline(list_stream, line)) {
            if (!line.empty()) {
                image_names.push_back(line);
            }
        }

        if (image_names.empty()) {
            std::cerr << "No images listed in " << list_file << std::endl;
            return 1;
        }

        std::cout << "========================================\n";
        std::cout << "ONNX Runtime Batch Inference\n";
        std::cout << "Model: " << model_path << "\n";
        std::cout << "Images: " << image_names.size() << "\n";
        std::cout << "========================================\n";

        std::vector<double> inference_times;
        inference_times.reserve(image_names.size());
        size_t processed = 0;
        size_t errors = 0;

        auto total_start = std::chrono::high_resolution_clock::now();
        for (size_t idx = 0; idx < image_names.size(); ++idx) {
            const std::filesystem::path image_path = image_root / image_names[idx];
            if (!std::filesystem::exists(image_path)) {
                std::cerr << "Missing image: " << image_path << std::endl;
                ++errors;
                continue;
            }

            cv::Mat original = cv::imread(image_path.string());
            if (original.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                ++errors;
                continue;
            }

            cv::Mat rgb = preprocess(original, runner.width(), runner.height());
            auto chw = to_chw(rgb);

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat prob = runner.infer(chw);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            inference_times.push_back(duration);

            save_results(original, prob, output_dir, image_path.stem().string());

            ++processed;
            if ((processed + errors) % 50 == 0 && !inference_times.empty()) {
                std::cout << "Progress: " << (processed + errors) << "/" << image_names.size()
                          << " | Successful: " << processed
                          << " | Avg time: "
                          << std::fixed << std::setprecision(2)
                          << (std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size())
                          << " ms" << std::endl;
            }
        }
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        double avg_time = inference_times.empty() ? 0.0 :
            std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();

        std::cout << "========================================\n";
        std::cout << "Batch Summary\n";
        std::cout << "Total images: " << image_names.size() << "\n";
        std::cout << "Processed: " << processed << "\n";
        std::cout << "Errors: " << errors << "\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " ms\n";
        std::cout << "Average inference: " << avg_time << " ms\n";
        if (processed > 0) {
            std::cout << "Total per-image (wall): " << (total_time / processed) << " ms\n";
        }
        std::cout << "Results saved to: " << output_dir << "\n";
        std::cout << "========================================\n";

        std::ofstream timing_out(timing_report);
        if (timing_out.is_open()) {
            timing_out << "ONNX Batch Inference Timing Results\n";
            timing_out << "====================================\n";
            timing_out << "Total images: " << image_names.size() << "\n";
            timing_out << "Successfully processed: " << processed << "\n";
            timing_out << "Errors: " << errors << "\n";
            timing_out << "Total processing time: " << total_time << " ms\n";
            timing_out << "Average inference time per image: " << avg_time << " ms\n";
            if (processed > 0) {
                timing_out << "Total processing time per image: " << (total_time / processed) << " ms\n";
            }
            timing_out << "\nDetailed per-image timing (ms):\n";
            for (size_t i = 0; i < inference_times.size(); ++i) {
                timing_out << "Image " << (i + 1) << ": " << inference_times[i] << "\n";
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
