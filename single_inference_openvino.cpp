#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
class OpenVinoRunner {
public:
    explicit OpenVinoRunner(const std::string& model_path) {
        // 创建OpenVINO Core
        ov::Core core;
        
        // 读取ONNX模型
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        
        // 获取输入输出信息
        if (model->inputs().empty() || model->outputs().empty()) {
            throw std::runtime_error("Model has no inputs or outputs");
        }
        
        auto input = model->input();
        auto output = model->output();
        
        input_name_ = input.get_any_name();
        output_name_ = output.get_any_name();
        
        // 获取输入形状
        ov::PartialShape input_pshape = input.get_partial_shape();
        if (input_pshape.rank().get_length() != 4) {
            throw std::runtime_error("Model input must be NCHW");
        }
        
        // 处理动态形状，设置固定维度
        channels_ = 3;  // 默认RGB
        height_ = 512;  // 默认尺寸
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
        
        // 如果有动态维度，使用reshape固定输入形状
        if (input_pshape.is_dynamic()) {
            std::cout << "Model has dynamic input shape, setting to [1, " 
                      << channels_ << ", " << height_ << ", " << width_ << "]" << std::endl;
            std::map<std::string, ov::PartialShape> input_shapes;
            input_shapes[input_name_] = ov::PartialShape{1, channels_, height_, width_};
            model->reshape(input_shapes);
        }
        
        std::cout << "Model input: " << input_name_ << " [1, " << channels_ 
                  << ", " << height_ << ", " << width_ << "]" << std::endl;
        std::cout << "Model output: " << output_name_ << std::endl;
        
        // 编译模型
        device_ = "CPU";
        compiled_model_ = core.compile_model(model, device_);
        std::cout << "Using OpenVINO (" << device_ << ")" << std::endl;
        infer_request_ = compiled_model_.create_infer_request();
    }

    cv::Mat infer(const std::vector<float>& input_data) {
        // 创建输入张量
        ov::Shape input_shape = {1, static_cast<size_t>(channels_), 
                                  static_cast<size_t>(height_), 
                                  static_cast<size_t>(width_)};
        ov::Tensor input_tensor(ov::element::f32, input_shape, 
                                const_cast<float*>(input_data.data()));
        
        // 设置输入
        infer_request_.set_input_tensor(input_tensor);
        
        // 执行推理
        infer_request_.infer();
        
        // 获取输出
        ov::Tensor output_tensor = infer_request_.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        
        if (output_shape.size() != 4) {
            throw std::runtime_error("Unexpected output shape");
        }
        
        const int out_h = static_cast<int>(output_shape[2]);
        const int out_w = static_cast<int>(output_shape[3]);
        const float* data = output_tensor.data<float>();
        
        // 转换为OpenCV Mat并应用sigmoid
        cv::Mat prob(out_h, out_w, CV_32F);
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                float raw = data[h * out_w + w];
                prob.at<float>(h, w) = 1.0f / (1.0f + std::exp(-raw));
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

cv::Mat preprocess(const cv::Mat& image, int target_width, int target_height) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width, target_height));
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

int main() {
    try {
        const std::string model_path = "D:/ertek_codebase/onnx_demo/models/model.onnx";
        const std::string image_path = "D:/ertek_data/scratch_data/images/Temp0027F_0_0_0_832.png";
        const std::string output_prefix = "openvino_cpu_inference";

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

        OpenVinoRunner runner(model_path);
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
        std::cout << "[OpenVINO - " << runner.device() << "] Inference time: " << duration << " ms" << std::endl;

        save_outputs(image, prob, output_prefix);
        std::cout << "Result saved: " << output_prefix << "_result.png" << std::endl;
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
