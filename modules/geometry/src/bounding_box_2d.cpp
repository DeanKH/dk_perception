#include <Eigen/Eigenvalues>
#include <cmath>
#include <dk_perception/geometry/bounding_box_2d.hpp>
#include <stdexcept>

namespace dklib::perception::geometry {

BoundingBox2D::BoundingBox2D() : center(Eigen::Vector2f::Zero()), size(Eigen::Vector2f::Zero()), rotation(0.0f) {}

BoundingBox2D BoundingBox2D::from_center_size(const Eigen::Vector2f& center, const Eigen::Vector2f& size,
                                              float rotation) {
  BoundingBox2D bbox;
  bbox.center = center;
  bbox.size = size;
  bbox.rotation = rotation;
  return bbox;
}

BoundingBox2D BoundingBox2D::from_corners(const Eigen::Vector2f& bottom_left, const Eigen::Vector2f& top_right,
                                          float rotation) {
  BoundingBox2D bbox;
  bbox.center = (bottom_left + top_right) / 2.0f;
  bbox.size = top_right - bottom_left;
  bbox.rotation = 0.0f;
  return bbox;
}

BoundingBox2D BoundingBox2D::from_axis_alined_points(const std::vector<Eigen::Vector2f>& points) {
  if (points.empty()) {
    throw std::runtime_error("Point list is empty");
  }

  Eigen::Vector2f min_pt = points[0];
  Eigen::Vector2f max_pt = points[0];

  for (const auto& pt : points) {
    min_pt = min_pt.cwiseMin(pt);
    max_pt = max_pt.cwiseMax(pt);
  }

  return from_corners(min_pt, max_pt);
}

BoundingBox2D BoundingBox2D::from_points_pca_aligned(const std::vector<Eigen::Vector2f>& points) {
  if (points.size() < 2) {
    throw std::runtime_error("At least two points are required for PCA alignment");
  }
  // Compute the centroid
  Eigen::Vector2f centroid(0.0f, 0.0f);
  for (const auto& pt : points) {
    centroid += pt;
  }
  centroid /= static_cast<float>(points.size());

  // Compute the covariance matrix
  Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
  for (const auto& pt : points) {
    Eigen::Vector2f centered = pt - centroid;
    covariance += centered * centered.transpose();
  }
  covariance /= static_cast<float>(points.size());
  // Perform eigen decomposition
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Eigen decomposition failed");
  }
  Eigen::Vector2f eigenvalues = solver.eigenvalues();
  Eigen::Matrix2f eigenvectors = solver.eigenvectors();
  // The principal component is the eigenvector with the largest eigenvalue
  Eigen::Vector2f principal_axis = eigenvectors.col(1);
  float angle = std::atan2(principal_axis.y(), principal_axis.x());
  // Rotate points to align with principal axis
  Eigen::Matrix2f rotation_matrix;
  rotation_matrix << cos(-angle), -sin(-angle), sin(-angle), cos(-angle);
  std::vector<Eigen::Vector2f> rotated_points;
  rotated_points.reserve(points.size());
  for (const auto& pt : points) {
    rotated_points.push_back(rotation_matrix * (pt - centroid));
  }

  // Compute axis-aligned bounding box in rotated frame
  Eigen::Vector2f min_pt = rotated_points[0];
  Eigen::Vector2f max_pt = rotated_points[0];
  for (const auto& pt : rotated_points) {
    min_pt = min_pt.cwiseMin(pt);
    max_pt = max_pt.cwiseMax(pt);
  }

  return from_center_size(centroid, Eigen::Vector2f(max_pt - min_pt), angle);
}

}  // namespace dklib::perception::geometry