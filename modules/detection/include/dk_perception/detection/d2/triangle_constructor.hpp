#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "dk_perception/dnn/deep_lsd.hpp"
#include "dk_perception/type/polygons.hpp"

namespace dklib::perception::detection::d2 {

class TriangleConstructorByLineIntersections {
 public:
  TriangleConstructorByLineIntersections(const dklib::dnn::DeepFastLineSegmentDetector::Ptr& line_segment_detector)
      : line_segment_detector_(line_segment_detector) {}
  ~TriangleConstructorByLineIntersections() = default;

  dklib::perception::type::Polygons<Eigen::Vector2f> construct(const cv::Mat& image);

 private:
  dklib::dnn::DeepFastLineSegmentDetector::Ptr line_segment_detector_;
};

struct IntersectionInfo {
  Eigen::Vector2f intersection_point;  // Intersection point of the two line segments
  float t1;                            // Parameter for the first line segment
  float t2;                            // Parameter for the second line segment
};

bool calcIntersection(const Eigen::Vector2f& l1_p1, const Eigen::Vector2f& l1_p2, const Eigen::Vector2f& l2_p1,
                      const Eigen::Vector2f& l2_p2, float& t1, float& t2) {
  Eigen::Vector2f l1 = (l1_p2 - l1_p1);
  Eigen::Vector2f l2 = (l2_p2 - l2_p1);

  // Calculate the denominator of the intersection formula
  float denominator = l1[0] * l2[1] - l1[1] * l2[0];

  // If denominator is close to zero, lines are parallel or collinear
  if (std::abs(denominator) < 1e-6) {
    return false;
  }

  // Calculate parameters t1 and t2
  Eigen::Vector2f diff = l2_p1 - l1_p1;
  t1 = (diff[0] * l2[1] - diff[1] * l2[0]) / denominator;
  t2 = (diff[0] * l1[1] - diff[1] * l1[0]) / denominator;

  // Check if the intersection point is within both line segments
  // t1 ∈ [0,1] means intersection is on first line segment
  // t2 ∈ [0,1] means intersection is on second line segment
  return (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1);
}

std::shared_ptr<IntersectionInfo> calcExtendedIntersection(
    const std::pair<Eigen::Vector2f, Eigen::Vector2f>& line_segment1,
    const std::pair<Eigen::Vector2f, Eigen::Vector2f>& line_segment2, float line_extension_ratio) {
  const Eigen::Vector2f l1_p1 = line_segment1.first;
  const Eigen::Vector2f l1_p2 = line_segment1.second;
  const Eigen::Vector2f l2_p1 = line_segment2.first;
  const Eigen::Vector2f l2_p2 = line_segment2.second;

  const Eigen::Vector2f l1 = l1_p2 - l1_p1;
  const Eigen::Vector2f l2 = l2_p2 - l2_p1;

  const Eigen::Vector2f l1_dir = l1.normalized();
  const Eigen::Vector2f l2_dir = l2.normalized();
  const Eigen::Vector2f l1_extended_p1 = l1_p1 - line_extension_ratio * l1;
  const Eigen::Vector2f l1_extended_p2 = l1_p2 + line_extension_ratio * l1;
  const Eigen::Vector2f l2_extended_p1 = l2_p1 - line_extension_ratio * l2;
  const Eigen::Vector2f l2_extended_p2 = l2_p2 + line_extension_ratio * l2;

  float extended_t1 = 0;
  float extended_t2 = 0;
  if (!calcIntersection(l1_extended_p1, l1_extended_p2, l2_extended_p1, l2_extended_p2, extended_t1, extended_t2)) {
    return nullptr;
  }
  std::cout << "Extended intersection found: t1 = " << extended_t1 << ", t2 = " << extended_t2 << std::endl;

  std::shared_ptr<IntersectionInfo> info = std::make_shared<IntersectionInfo>();

  // Calculate the intersection point
  info->intersection_point = l1_extended_p1 + extended_t1 * (l1_extended_p2 - l1_extended_p1);

  // extend_t1から元の線分に対するパラメータt1を計算する．t1は0~1の範囲外でも構わない
  // (intersection_point - l1)とl1_dirの方向が同じか
  const Eigen::Vector2f l1_e = info->intersection_point - l1;
  if (l1_e.normalized().dot(l1_dir) > 0) {
    // t1 > 0
    info->t1 = l1_e.norm() / l1.norm();
  } else {
    // t1 < 0
    info->t1 = -l1_e.norm() / l1.norm();
  }

  // extend_t2から元の線分に対するパラメータt2を計算する．t2は0~1の範囲外でも構わない
  // (intersection_point - l2)とl2_dirの方向が同じか
  const Eigen::Vector2f l2_e = info->intersection_point - l2;
  if (l2_e.normalized().dot(l2_dir) > 0) {
    // t2 > 0
    info->t2 = l2_e.norm() / l2.norm();
  } else {
    // t2 < 0
    info->t2 = -l2_e.norm() / l2.norm();
  }

  return info;
}

std::map<std::pair<size_t, size_t>, std::shared_ptr<IntersectionInfo>> findLineIntersections(
    const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>& line_segments,
    float right_angle_tolerance_deg = 10.0f, float line_extension_ratio = 10.0f) {
  std::map<std::pair<size_t, size_t>, std::shared_ptr<IntersectionInfo>> intersections;

  for (size_t i = 0; i < line_segments.size(); ++i) {
    for (size_t j = i + 1; j < line_segments.size(); ++j) {
      const auto& line_segment1 = line_segments[i];
      const auto& line_segment2 = line_segments[j];
      // calc angle between two line segments
      double angle = std::acos((line_segment1.second - line_segment1.first)
                                   .normalized()
                                   .dot((line_segment2.second - line_segment2.first).normalized()));
      if (angle > M_PI / 2.0) {
        angle = M_PI - angle;  // Use the smaller angle
      }
      // Check if the angle is within the right angle tolerance
      if (std::fabs(angle * 180.0 / M_PI - 90.0) > right_angle_tolerance_deg) {
        continue;
      }

      auto intersection_info = calcExtendedIntersection(line_segment1, line_segment2, line_extension_ratio);
      if (intersection_info) {
        intersections[{i, j}] = intersection_info;
      }
    }
  }

  return intersections;
}

}  // namespace dklib::perception::detection::d2