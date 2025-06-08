#pragma once
#include <opencv2/core.hpp>

namespace dklib::perception::detection::d2 {

class TriangleDetection {
 public:
  TriangleDetection();
  ~TriangleDetection();

  void detect(const cv::Mat& image);
};

}  // namespace dklib::perception::detection::d2