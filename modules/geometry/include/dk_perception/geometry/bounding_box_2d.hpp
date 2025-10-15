#pragma once
#include <Eigen/Core>
#include <vector>

namespace dklib::perception::geometry {
struct BoundingBox2D {
  Eigen::Vector2f center;
  Eigen::Vector2f size;
  float rotation;  // in radians

  BoundingBox2D();

  static BoundingBox2D from_center_size(const Eigen::Vector2f& center, const Eigen::Vector2f& size,
                                        float rotation = 0.0);

  static BoundingBox2D from_corners(const Eigen::Vector2f& bottom_left, const Eigen::Vector2f& top_right,
                                    float rotation = 0.0);

  static BoundingBox2D from_axis_alined_points(const std::vector<Eigen::Vector2f>& points);

  static BoundingBox2D from_points_pca_aligned(const std::vector<Eigen::Vector2f>& points);
};

}  // namespace dklib::perception::geometry