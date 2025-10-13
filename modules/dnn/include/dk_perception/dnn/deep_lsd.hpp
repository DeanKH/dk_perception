#pragma once
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>

// #include "dk_perception/common/macros.hpp"
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

/**
 * @brief DeepLSD + Fast Line Detector
 */
class DeepFastLineSegmentDetector {
 public:
  DeepFastLineSegmentDetector(
      const std::filesystem::path& model_path, InferenceDevice inference_device = InferenceDevice::kCUDA,
      DeepLsdAttractionField::InputSize input_size = DeepLsdAttractionField::InputSize::kInputSize1280) {
    // Initialize DeepLsdAttractionField
    deep_lsd_ = std::make_unique<DeepLsdAttractionField>(model_path, inference_device, input_size);

    int length_threshold = 40;
    float distance_threshold = 1.414213562f * 8;
    // disable Canny edge detection
    int canny_aperture_size = 3;
    double canny_th1 = 100.0;
    double canny_th2 = 100.0;
    bool do_merge = true;
    line_detector_ = cv::ximgproc::createFastLineDetector(length_threshold, distance_threshold, canny_th1, canny_th2,
                                                          canny_aperture_size, do_merge);
  }

  void detect(const cv::Mat& image, std::vector<cv::Vec4f>& lines) {
    auto result = deep_lsd_->inference(image);
    cv::Mat df_norm;
    cv::normalize(result.df_norm, df_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

    // TODO: disable canny by parameter
    line_detector_->detect(df_norm, lines);

    // TODO: refine by AF
  }

 private:
  std::unique_ptr<DeepLsdAttractionField> deep_lsd_;
  cv::Ptr<cv::ximgproc::FastLineDetector> line_detector_;

 public:
  using Ptr = std::shared_ptr<DeepFastLineSegmentDetector>;
};
}  // namespace dklib::dnn