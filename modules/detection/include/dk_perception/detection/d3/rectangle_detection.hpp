#pragma once
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/types.h>

#include <concepts>
#include <cstddef>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>
#include <dk_perception/type/pointcloud/rgbd_type.hpp>
#include <opencv2/core.hpp>

namespace dklib::perception::detection::d3 {

/// 直角三角形
struct RightAngleTriangle {
  std::array<size_t, 3> vertex_indices;  // Indices of the triangle vertices
  size_t right_angle_vertex_index;       // Index of the vertex at the right angle
  Eigen::Vector3f normal;

  size_t operator[](size_t index) const {
    if (index < 3) {
      return vertex_indices[index];
    } else {
      throw std::out_of_range("Index out of range for RightAngleTriangle");
    }
  }

  size_t& operator[](size_t index) {
    if (index < 3) {
      return vertex_indices[index];
    } else {
      throw std::out_of_range("Index out of range for RightAngleTriangle");
    }
  }
};

template <typename PointT2>
void excludeInvalidTriangles(const std::vector<Eigen::Vector3f>& points, std::vector<RightAngleTriangle>& triangles,
                             const typename pcl::search::KdTree<PointT2>::Ptr& kdtree,
                             const size_t num_of_divisions = 10, const double distance_threshold = 0.02,
                             const double invalid_ratio = 0.1) {
  /// 1. 直角三角形の辺の内，直角を構成する線分2本を取得
  ///    (e.g. p1, p2, p3のうちp1とp2を結ぶ線分とp1とp3を結ぶ線分)
  /// 2. 3D線分を均等にnum_of_divisions分割(e.g. 10)
  /// 3. 線分の分割点の最近傍点を取得. 始点と終点は除く
  /// 4. 分割点の最近傍点までの距離が閾値を超えている場合がM%未満の線分を削除
  std::vector<size_t> indices_to_remove;
  for (size_t i = 0; i < triangles.size(); ++i) {
    const auto& triangle = triangles[i];
    const Eigen::Vector3f& p1 = points[triangle.vertex_indices[0]];
    const Eigen::Vector3f& p2 = points[triangle.vertex_indices[1]];
    const Eigen::Vector3f& p3 = points[triangle.vertex_indices[2]];

    // triangle.vertex_indices から triangle.right_angle_vertex_indexを除いた2点をopposite_side_indicesに格納
    std::array<std::pair<Eigen::Vector3f, Eigen::Vector3f>, 2> segments;
    if (triangle.right_angle_vertex_index == triangle.vertex_indices[0]) {
      segments = {std::make_pair(p1, p2), std::make_pair(p1, p3)};
    } else if (triangle.right_angle_vertex_index == triangle.vertex_indices[1]) {
      segments = {std::make_pair(p2, p1), std::make_pair(p2, p3)};
    } else {
      segments = {std::make_pair(p3, p1), std::make_pair(p3, p2)};
    }

    // 線分を均等に分割
    std::array<std::vector<double>, 2> segment_nearest_points;
    for (size_t j = 0; j < segments.size(); ++j) {
      const auto& [start, end] = segments[j];
      size_t count_invalid = 0;
      for (size_t k = 1; k < num_of_divisions; ++k) {
        Eigen::Vector3f point = start + (end - start) * (static_cast<double>(k) / num_of_divisions);
        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;
        PointT2 point_t(point.x(), point.y(), point.z());
        kdtree->nearestKSearch(point_t, 1, k_indices, k_sqr_distances);
        if (k_indices.empty() || k_sqr_distances[0] > distance_threshold * distance_threshold) {
          count_invalid++;
        }
      }
      if (static_cast<double>(count_invalid) / (num_of_divisions - 1) > invalid_ratio) {
        indices_to_remove.push_back(i);
        break;  // この三角形は削除対象
      }
    }
  }
  // indices_to_removeを逆順にソートして削除
  std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
  for (size_t index : indices_to_remove) {
    if (index < triangles.size()) {
      triangles.erase(triangles.begin() + index);
    }
  }
}

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
  std::vector<RightAngleTriangle> construct(const std::vector<PointT>& points) const {
    const Eigen::Vector3f camera_optical_minus(0, 0, -1);  // Normal vector for the ground plane at camera

    std::vector<RightAngleTriangle> triangles;
    for (size_t i = 0; i < points.size(); ++i) {
      for (size_t j = 1 + 1; j < points.size(); ++j) {
        for (size_t k = j + 1; k < points.size(); ++k) {
          // if (i == j || j == k || i == k) {
          //   continue;  // Skip if any two indices are the same
          // }
          auto right_angle_index = isRightAngleTriangle(points[i], points[j], points[k]);
          if (right_angle_index != -1) {
            RightAngleTriangle triangle{.vertex_indices = {i, j, k},
                                        .right_angle_vertex_index = right_angle_index,
                                        .normal = (points[j] - points[i]).cross(points[k] - points[i])};
            triangle.normal.normalize();

            double normal_angle = std::acos(triangle.normal.dot(camera_optical_minus)) * 180.0 / M_PI;
            if (!(normal_angle < normal_tolerance_deg_)) {
              continue;
            }

            triangles.emplace_back(triangle);
          }
        }
      }
    }
    return triangles;
  }

 private:
  ptrdiff_t isRightAngleTriangle(const PointT& p1, const PointT& p2, const PointT& p3) const {
    double d1 = (p2 - p1).norm();
    double d2 = (p3 - p2).norm();
    double d3 = (p1 - p3).norm();

    // Check if the triangle is a right-angle triangle
    if (d1 < min_distance_ || d2 < min_distance_ || d3 < min_distance_) {
      return -1;
    }

    // それぞれの点同士を結ぶ線分同士のなす角が許容範囲内かをチェック
    double angle1 = angleBetween(p1, p2, p3);
    double angle2 = angleBetween(p2, p3, p1);
    double angle3 = angleBetween(p3, p1, p2);

    if (std::abs(angle1 - 90.0) > angle_tolerance_deg_ && std::abs(angle2 - 90.0) > angle_tolerance_deg_ &&
        std::abs(angle3 - 90.0) > angle_tolerance_deg_) {
      return -1;
    }

    // 直角三角形の頂点のインデックスを返す
    if (std::abs(angle1 - 90.0) < angle_tolerance_deg_) {
      return 0;
    } else if (std::abs(angle2 - 90.0) < angle_tolerance_deg_) {
      return 1;
    } else if (std::abs(angle3 - 90.0) < angle_tolerance_deg_) {
      return 2;
    }
    return -1;  // 直角三角形ではない
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