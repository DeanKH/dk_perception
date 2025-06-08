#pragma once

#include <dk_perception/dnn/superpoint.hpp>
#include <filesystem>

namespace dklib::perception::detection::d2 {

class NodeExtractorFromImageFeatures {
 public:
  NodeExtractorFromImageFeatures(const std::shared_ptr<dklib::experimental::SuperPoint>& superpoint,
                                 const std::string& vocab_dictionary_path);

  std::vector<cv::Point2f> extract(const cv::Mat& image);
  void setSimilarityThreshold(double threshold) { similarity_threshold_ = threshold; }
  void setROI(const cv::Rect& region_of_interest) { roi = region_of_interest; }
  void setMinScore(double min_score) { min_score_ = min_score; }

 private:
  std::shared_ptr<dklib::experimental::SuperPoint> superpoint_;

  cv::Mat vocab_;
  double similarity_threshold_ = 0.5;
  double min_score_ = 0.0005;
  cv::Rect roi = cv::Rect(0, 0, 0, 0);  // 初期化用
};
}  // namespace dklib::perception::detection::d2