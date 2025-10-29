#pragma once
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vector>

namespace dklib::perception::type {

struct Polygon {
  std::vector<size_t> indices;
};

template <typename PointT = cv::Point2f>
struct Polygons {
  std::vector<Polygon> polygons;
  std::vector<PointT> points;
};

}  // namespace dklib::perception::type