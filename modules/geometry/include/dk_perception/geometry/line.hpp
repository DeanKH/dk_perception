#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace dklib::perception::geometry {
template <typename PointT>
struct Line {
  PointT start;  ///< Start point of the line.
  PointT end;    ///< End point of the line.
};

template <typename PointT, typename VectorT = PointT>
struct LineRef {
  const PointT& start;  ///< Reference to the start point of the line.
  const PointT& end;    ///< Reference to the end point of the line.

  VectorT vector() const {
    return end - start;  ///< Returns the vector representation of the line.
  }

  float angle(const LineRef<PointT, VectorT>& other) const {
    const PointT l1_vector = vector();
    const PointT l2_vector = other.vector();

    // Calculate the angle between the two line segments using the dot product
    float dot_product = l1_vector.dot(l2_vector);
    float magnitude_product = l1_vector.norm() * l2_vector.norm();
    if (magnitude_product == 0) {
      return std::numeric_limits<float>::quiet_NaN();  // Avoid division by zero
    }
    return std::acos(dot_product / magnitude_product);
  }

  PointT at(float t) const {
    return start + t * vector();  ///< Returns the point on the line at parameter t.
  }
};

struct LineIndices {
  size_t start_idx;
  size_t end_idx;

  template <typename PointT>
  LineRef<PointT> toLineRef(const std::vector<PointT>& points) const {
    return LineRef<PointT>{points[start_idx], points[end_idx]};
  }

  template <typename PointT>
  float angle(const LineIndices& other, const std::vector<PointT>& points) const {
    const LineRef<PointT> l1 = toLineRef(points);
    const LineRef<PointT> l2 = other.toLineRef(points);

    return l1.angle(l2);  ///< Returns the angle between this line and another line.
  }

  size_t far_idx(float t) const {
    return t < 0.5 ? end_idx : start_idx;  ///< Returns the index of the point at parameter t.
  }

  size_t near_idx(float t) const {
    return t < 0.5 ? start_idx : end_idx;  ///< Returns the index of the point at parameter t.
  }
};

template <typename PointT>
std::pair<std::vector<LineIndices>, std::vector<PointT>> extractLineSegments(const std::vector<Line<PointT>>& lines,
                                                                             bool merge_points = false,
                                                                             float merge_distance = 0.001f) {
  assert(!merge_points);  // near points Merging is not implemented yet."

  std::vector<LineIndices> line_indices_list;
  std::vector<PointT> points;

  line_indices_list.reserve(lines.size());
  points.reserve(lines.size() * 2);
  for (const auto& line : lines) {
    line_indices_list.emplace_back(points.size(), points.size() + 1);
    points.push_back(line.start);
    points.push_back(line.end);
  }

  points.shrink_to_fit();
  return {line_indices_list, points};
}

}  // namespace dklib::perception::geometry