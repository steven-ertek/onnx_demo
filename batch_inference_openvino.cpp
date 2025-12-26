#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class OpenVinoRunner {
public:
    explicit OpenVinoRunner(const std::string& model_path) {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        
        if (model->inputs().empty() || model->outputs().empty()) {
            throw std::runtime_error("Model has no inputs or outputs");
        }
        
        auto input = model->input();
        auto output = model->output();
        
        input_name_ = input.get_any_name();
        output_name_ = output.get_any_name();
        
        ov::PartialShape input_pshape = input.get_partial_shape();
        if (input_pshape.rank().get_length() != 4) {
            throw std::runtime_error("Model input must be NCHW");
        }
        
        channels_ = 3;
        height_ = 512;
        width_ = 512;
        
        if (input_pshape[1].is_static()) {
            channels_ = static_cast<int>(input_pshape[1].get_length());
        }
        if (input_pshape[2].is_static()) {
            height_ = static_cast<int>(input_pshape[2].get_length());
        }
        if (input_pshape[3].is_static()) {
            width_ = static_cast<int>(input_pshape[3].get_length());
        }
        
        if (input_pshape.is_dynamic()) {
            std::map<std::string, ov::PartialShape> input_shapes;
            input_shapes[input_name_] = ov::PartialShape{1, channels_, height_, width_};
            model->reshape(input_shapes);
        }
        
        device_ = "CPU";
        compiled_model_ = core.compile_model(model, device_);
        infer_request_ = compiled_model_.create_infer_request();
    }

    cv::Mat infer(const std::vector<float>& input_data) {
        ov::Shape input_shape = {1, static_cast<size_t>(channels_), 
                                  static_cast<size_t>(height_), 
                                  static_cast<size_t>(width_)};
        ov::Tensor input_tensor(ov::element::f32, input_shape, 
                                const_cast<float*>(input_data.data()));
        
        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();
        
        ov::Tensor output_tensor = infer_request_.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        
        if (output_shape.size() != 4) {
            throw std::runtime_error("Unexpected output shape");
        }
        
        const int out_h = static_cast<int>(output_shape[2]);
        const int out_w = static_cast<int>(output_shape[3]);
        const float* data = output_tensor.data<float>();
        
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
    std::string device() const { return device_; }

private:
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    std::string input_name_;
    std::string output_name_;
    std::string device_;
    int channels_{0};
    int height_{0};
    int width_{0};
};

cv::Mat preprocess(const cv::Mat& image, int target_w, int target_h) {
    // 先转换为 RGB，再 resize（与 PyTorch 版本一致）
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(target_w, target_h));
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

    // cv::imwrite((out_dir / (stem + "_binary.png")).string(), binary_u8);
    cv::imwrite((out_dir / (stem + "_combined.png")).string(), combined);
}
} // namespace

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>        Path to ONNX model (default: models/model.onnx)\n";
    std::cout << "  --images <path>       Root directory containing images (default: D:/ertek_data/scratch_data/images)\n";
    std::cout << "  --list <path>         Text file with image names (default: test.txt)\n";
    std::cout << "  --output <path>       Output directory for results (default: batch_results_openvino)\n";
    std::cout << "  --batch <size>        Batch size for inference (default: 8)\n";
    std::cout << "  --timing <path>       Timing report file (default: openvino_batch_timing.txt)\n";
    std::cout << "  --save <0|1>          Save prediction images (default: 1, 0=no save)\n";
    std::cout << "  --help                Show this help message\n";
}

int main(int argc, char* argv[]) {
    // 默认值
    std::filesystem::path model_path = "models/model.onnx";
    std::filesystem::path image_root = "D:/ertek_data/scratch_data/images";
    std::filesystem::path list_file = "test.txt";
    std::filesystem::path output_dir = "batch_results_openvino";
    std::filesystem::path timing_report = "openvino_batch_timing.txt";
    int batch_size = 8;
    bool enable_save = true;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--images" && i + 1 < argc) {
            image_root = argv[++i];
        } else if (arg == "--list" && i + 1 < argc) {
            list_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--timing" && i + 1 < argc) {
            timing_report = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            enable_save = (std::stoi(argv[++i]) != 0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model not found: " << model_path << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(list_file)) {
            std::cerr << "Test list not found: " << list_file << std::endl;
            return 1;
        }

        std::filesystem::create_directories(output_dir);
        OpenVinoRunner runner(model_path.string());

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
        std::cout << "OpenVINO Batch Inference\n";
        std::cout << "Device: " << runner.device() << "\n";
        std::cout << "Model: " << model_path << "\n";
        std::cout << "Images: " << image_names.size() << "\n";
        std::cout << "========================================\n";

        std::vector<double> inference_times;
        inference_times.reserve(image_names.size());
        size_t processed = 0;
        size_t errors = 0;

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

            if (enable_save) {
                save_results(original, prob, output_dir, image_path.stem().string());
            }

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

        double avg_time = inference_times.empty() ? 0.0 :
            std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
        double total_inference_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);

        std::cout << "========================================\n";
        std::cout << "Batch Summary\n";
        std::cout << "Device: " << runner.device() << "\n";
        std::cout << "Total images: " << image_names.size() << "\n";
        std::cout << "Processed: " << processed << "\n";
        std::cout << "Errors: " << errors << "\n";
        std::cout << "Total inference time: " << std::fixed << std::setprecision(2) << total_inference_time << " ms\n";
        std::cout << "Average inference time: " << avg_time << " ms\n";
        std::cout << "Results saved to: " << output_dir << "\n";
        std::cout << "========================================\n";

        std::ofstream timing_out(timing_report);
        if (timing_out.is_open()) {
            timing_out << "OpenVINO Batch Inference Timing Results\n";
            timing_out << "========================================\n";
            timing_out << "Device: " << runner.device() << "\n";
            timing_out << "Total images: " << image_names.size() << "\n";
            timing_out << "Successfully processed: " << processed << "\n";
            timing_out << "Errors: " << errors << "\n";
            timing_out << "Total inference time: " << total_inference_time << " ms\n";
            timing_out << "Average inference time per image: " << avg_time << " ms\n";
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
