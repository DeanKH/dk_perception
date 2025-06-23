#pragma once
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "dk_perception/dnn/common.hpp"

namespace dklib::dnn {
class DeepLsdAttractionField {
 public:
  enum class InputSize : uint32_t {
    kInputSizeFlexible = 0,
    kInputSize1280 = 1280,
  };

  struct Result {
    cv::Mat df_norm;
    cv::Mat df;
    cv::Mat line_level;
  };

  DeepLsdAttractionField(const std::filesystem::path& model_path,
                         InferenceDevice inference_device = InferenceDevice::kCUDA,
                         InputSize input_size = InputSize::kInputSize1280);
  ~DeepLsdAttractionField() = default;

  Result inference(const cv::Mat& input);

 private:
  cv::Mat preprocessImage(const cv::Mat& image) {
    assert(input_size_ == InputSize::kInputSize1280);
    assert(image.size().width == 1280 && image.size().height == 720);
    assert(image.depth() == CV_8U);
    assert(image.channels() == 3);

    // normalize the image to [0, 1] range
    cv::Mat normalize_image;
    image.convertTo(normalize_image, CV_32F, 1.0 / 255.0);
    return normalize_image;
  }

  Result postProcessImage(const std::vector<Ort::Value>& tensor);
  std::vector<Ort::Value> inferenceImpl(const cv::Mat& preprocessed_image);
  void configureInOutNodes();

  InputSize input_size_ = InputSize::kInputSize1280;
  InferenceDevice inference_device_ = InferenceDevice::kCUDA;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<char*> input_node_names_;
  std::vector<std::vector<int64_t>> input_node_shapes_;
  std::vector<char*> output_node_names_;
  std::vector<std::vector<int64_t>> output_node_shapes_;
};
}  // namespace dklib::dnn