#pragma once

#include <Eigen/Core>
#include <dk_perception/geometry/line.hpp>

namespace dklib::perception::geometry {
template <typename PointT>
struct IntersectionInfo {
  PointT intersection_point;  ///< Intersection point of the two line segments
  float t1;                   ///< Parameter for the first line segment
  float t2;                   ///< Parameter for the second line segment
};

float crossMagnitude(const Eigen::Vector2f& vec1, const Eigen::Vector2f& vec2) {
  return vec1.x() * vec2.y() - vec1.y() * vec2.x();
}

template <typename PointT = Eigen::Vector2f, typename VectorT = PointT>
bool calcIntersection(const std::vector<PointT>& points, const LineIndices& line1, const LineIndices& line2, float& t1,
                      float& t2) {
  const LineRef<PointT> l1 = line1.toLineRef(points);
  const LineRef<PointT> l2 = line2.toLineRef(points);
  const PointT l1_vector = l1.vector();
  const PointT l2_vector = l2.vector();
  float denominator = crossMagnitude(l1_vector, l2_vector);

  if (std::abs(denominator) < 1e-6) {
    // Lines are parallel or collinear
    return false;
  }

  VectorT diff = l2.start - l1.start;
  t1 = (crossMagnitude(diff, l2_vector)) / denominator;
  t2 = (crossMagnitude(diff, l1_vector)) / denominator;

  // Check if the intersection point is within both line segments
  // t1 ∈ [0,1] means intersection is on first line segment
  // t2 ∈ [0,1] means intersection is on second line segment
  return (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1);
}

}  // namespace dklib::perception::geometry