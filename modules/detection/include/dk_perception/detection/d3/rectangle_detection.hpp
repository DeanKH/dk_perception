#pragma once
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>

#include <concepts>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>
#include <dk_perception/type/pointcloud/rgbd_type.hpp>
#include <opencv2/core.hpp>

namespace dklib::perception::detection::d3 {

template <class T>
concept MeshReconstructable = requires(T& x, pcl::PolygonMesh& mesh) {
  x.reconstruct(mesh);
};

template <MeshReconstructable T>
class RectangleDetection {
 public:
  RectangleDetection() {}
  ~RectangleDetection() {}

  void detect(const dklib::perception::type::pointcloud::DepthImageSet& rgbd) {
    dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<
        dklib::perception::type::pointcloud::DepthImageSet>
        accessor(rgbd);
  }
};

template <typename PointT = Eigen::Vector3f>
class RightAngleTriangleConstructor {
 public:
  RightAngleTriangleConstructor() {}
  ~RightAngleTriangleConstructor() {}

  // Constructs a right-angle triangle from three points
  std::vector<std::array<size_t, 3>> construct(const std::vector<PointT>& points) const {
    std::vector<std::array<size_t, 3>> vertex_index_pairs;
    for (size_t i = 0; i < points.size(); ++i) {
      for (size_t j = 1 + 1; j < points.size(); ++j) {
        for (size_t k = j + 1; k < points.size(); ++k) {
          // if (i == j || j == k || i == k) {
          //   continue;  // Skip if any two indices are the same
          // }
          if (isRightAngleTriangle(points[i], points[j], points[k])) {
            vertex_index_pairs.push_back({i, j, k});
          }
        }
      }
    }
    return vertex_index_pairs;
  }

 private:
  bool isRightAngleTriangle(const PointT& p1, const PointT& p2, const PointT& p3) const {
    double d1 = (p2 - p1).norm();
    double d2 = (p3 - p2).norm();
    double d3 = (p1 - p3).norm();

    // Check if the triangle is a right-angle triangle
    if (d1 < min_distance_ || d2 < min_distance_ || d3 < min_distance_) {
      return false;
    }

    // それぞれの点同士を結ぶ線分同士のなす角が許容範囲内かをチェック
    double angle1 = angleBetween(p1, p2, p3);
    double angle2 = angleBetween(p2, p3, p1);
    double angle3 = angleBetween(p3, p1, p2);

    if (std::abs(angle1 - 90.0) > angle_tolerance_deg_ && std::abs(angle2 - 90.0) > angle_tolerance_deg_ &&
        std::abs(angle3 - 90.0) > angle_tolerance_deg_) {
      return false;
    }

    Eigen::Vector3f normal1 = (p2 - p1).cross(p3 - p1);
    normal1.normalize();
    Eigen::Vector3f normal2(0, 0, -1);                                     // Normal vector for the ground plane
    double normal_angle = std::acos(normal1.dot(normal2)) * 180.0 / M_PI;  // Convert to degrees
    return normal_angle < normal_tolerance_deg_;
  }

  double angleBetween(const PointT& p1, const PointT& p2, const PointT& p3) const {
    Eigen::Vector3f v1 = p2 - p1;
    Eigen::Vector3f v2 = p3 - p1;
    double dot_product = v1.dot(v2);
    double magnitude_v1 = v1.norm();
    double magnitude_v2 = v2.norm();
    double cos_angle = dot_product / (magnitude_v1 * magnitude_v2);
    return std::acos(cos_angle) * 180.0 / M_PI;  // Convert to degrees
  }

  double min_distance_{0.1};
  double angle_tolerance_deg_{5.0};
  double normal_tolerance_deg_{10.0};  // tolerance for normal angle between triangle planes and (0,0,-1)
};
}  // namespace dklib::perception::detection::d3