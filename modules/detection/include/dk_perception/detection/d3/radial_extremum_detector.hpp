#pragma once
#include <pcl/PointIndices.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>

#include <dk_perception/pcproc/project_xz_plane.hpp>
#include <dk_perception/pcproc/radial_splitter.hpp>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>

namespace dklib::perception::detection::d3 {

/**
 * @brief 点群から最小値を計算する最も単純な方法.
 *
 * @tparam PointT
 * @param pc
 * @param transform_matrix
 * @return PointT
 */
template <typename PointT>
PointT calcMinimumPoint(const typename pcl::PointCloud<PointT>::Ptr& pc, const Eigen::Affine3f& transform_matrix) {
  typename pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>());
  dklib::perception::pcproc::projectXZPlane<PointT>(pc, transform_matrix, projected);
  // calc min z point
  PointT min_z_it = *std::min_element(projected->points.begin(), projected->points.end(),
                                      [](const auto& a, const auto& b) { return a.z < b.z; });

  Eigen::Affine3f inverse_transform = transform_matrix.inverse();
  min_z_it = pcl::transformPoint(min_z_it, inverse_transform);

  return min_z_it;
}

template <typename PointT>
class RadialExtremumDetector {
 public:
  RadialExtremumDetector(const pcproc::RadialSplitter<PointT>& splitter) : splitter_{splitter} {}

  typename pcl::PointCloud<PointT>::Ptr execute() {
    std::vector<float> angles;
    std::vector<typename pcl::PointCloud<PointT>::Ptr> points_list;
    splitter_.detect(angles, points_list);

    assert(angles.size() == points_list.size());

    typename pcl::PointCloud<PointT>::Ptr min_points(new pcl::PointCloud<PointT>());
    min_points->reserve(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
      const auto& subset = points_list[i];
      const Eigen::AngleAxisf rotation(M_PI * -angles[i] / 180.0f, splitter_.getAxis());
      const Eigen::Affine3f transform_matrix = rotation * Eigen::Translation3f(splitter_.getCenter());

      // TODO(deankh): Implement robust extremum point search
      PointT min_z_it = calcMinimumPoint<PointT>(subset, transform_matrix);
      min_points->points.push_back(min_z_it);
    }

    // 隣接するmin_points間の距離の分散が3σを超える点を除去
    {
      std::vector<float> distances;
      for (size_t i = 0; i < min_points->size(); ++i) {
        const auto& p1 = min_points->points[i];
        const auto& p2 = min_points->points[(i + 1) % min_points->size()];
        float distance = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
        distances.push_back(distance);
      }

      float mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
      float sq_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.0f);
      float stdev = std::sqrt(sq_sum / distances.size() - mean * mean);

      typename pcl::PointCloud<PointT>::Ptr filtered_min_points(new pcl::PointCloud<PointT>());
      for (size_t i = 0; i < min_points->size(); ++i) {
        if (std::abs(distances[i] - mean) <= stdev) {
          filtered_min_points->points.push_back(min_points->points[i]);
        }
      }
      min_points = filtered_min_points;
    }
    return min_points;
  }

 private:
  const pcproc::RadialSplitter<PointT>& splitter_;
};

}  // namespace dklib::perception::detection::d3