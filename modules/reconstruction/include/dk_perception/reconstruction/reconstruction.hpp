#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <voxblox/core/common.h>
#include <voxblox/core/esdf_map.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox/integrator/tsdf_integrator.h>

#include <dk_perception/geometry/bounding_box_3d.hpp>
#include <dk_perception/reconstruction/voxblox_pcl_conversion.hpp>

namespace dklib::perception::reconstruction {
template <typename PointT>
std::pair<voxblox::Pointcloud, voxblox::Colors> convertPointCloud(const pcl::PointCloud<PointT>& cloud) {
  voxblox::Pointcloud pointcloud;
  voxblox::Colors colors;
  pointcloud.reserve(cloud.size());
  colors.reserve(cloud.size());

  for (const auto& point : cloud.points) {
    pointcloud.emplace_back(point.x, point.y, point.z);
    colors.emplace_back(point.r, point.g, point.b);
  }

  return {pointcloud, colors};
}

class BoxInteriorReconstructor {
 public:
  // BoxInteriorReconstructor();
  BoxInteriorReconstructor(const geometry::BoundingBox3D& box, const float voxel_size);
  void reconstruct();

  template <typename PointT>
  void update(const pcl::PointCloud<PointT>& cloud, const Eigen::Matrix4f& cloud2boxbase_transform);

  const voxblox::Layer<voxblox::TsdfVoxel>& getTsdfLayer() const { return tsdf_map_->getTsdfLayer(); }

  pcl::PointCloud<pcl::PointXYZI> getSdfVoxelInBox() const {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    voxblox::createDistancePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(), &cloud);

    // filter by box
    pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
    filtered_cloud.reserve(cloud.size());
    const double voxel_half_size = voxel_size_ * 0.5;
    for (const auto& point : cloud.points) {
      const bool is_inside = [&]() {
        Eigen::Vector3d p(point.x, point.y, point.z);
        if (p.z() < 0.0) return false;
        if (p.x() < (box_.size.x() * -0.5 - voxel_half_size) || p.x() > (box_.size.x() * 0.5 + voxel_half_size))
          return false;
        if (p.y() < (box_.size.y() * -0.5 - voxel_half_size) || p.y() > (box_.size.y() * 0.5 + voxel_half_size))
          return false;

        return true;
      }();
      if (is_inside) {
        filtered_cloud.push_back(point);
      }
    }
    return filtered_cloud;
  }

  pcl::PointCloud<pcl::PointXYZI> getEsdfVoxelInBox() const {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    voxblox::createDistancePointcloudFromEsdfLayer(esdf_map_->getEsdfLayer(), &cloud);

    // filter by box
    pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
    filtered_cloud.reserve(cloud.size());
    const double voxel_half_size = voxel_size_ * 0.5;
    for (const auto& point : cloud.points) {
      const bool is_inside = [&]() {
        Eigen::Vector3d p(point.x, point.y, point.z);
        if (p.z() < 0.0) return false;
        if (p.x() < (box_.size.x() * -0.5 - voxel_half_size) || p.x() > (box_.size.x() * 0.5 + voxel_half_size))
          return false;
        if (p.y() < (box_.size.y() * -0.5 - voxel_half_size) || p.y() > (box_.size.y() * 0.5 + voxel_half_size))
          return false;

        return true;
      }();
      if (is_inside) {
        filtered_cloud.push_back(point);
      }
    }
    return filtered_cloud;
  }

 private:
  geometry::BoundingBox3D box_;
  float voxel_size_ = 0.02f;

  std::shared_ptr<voxblox::TsdfMap> tsdf_map_;
  voxblox::TsdfIntegratorBase::Ptr integrator_;

  std::shared_ptr<voxblox::EsdfMap> esdf_map_;
  std::unique_ptr<voxblox::EsdfIntegrator> esdf_integrator_;
  float box_max_half_size_;
};
}  // namespace dklib::perception::reconstruction