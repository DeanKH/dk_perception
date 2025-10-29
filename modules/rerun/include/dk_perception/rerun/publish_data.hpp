#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include <gtsam_app/geoms.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <voxblox/core/color.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cassert>
#include <dk_perception/geometry.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/boxes3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/recording_stream.hpp>
#include <vector>

#include "rerun/archetypes/boxes3d.hpp"
#include "rerun/archetypes/points3d.hpp"
#include "rerun/archetypes/transform3d.hpp"
#include "rerun/rotation3d.hpp"

namespace dklib::perception::publisher {
std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const Eigen::Isometry3d& transform);

std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const dklib::perception::geometry::BoundingBox3D& bbox,
                        const std::array<uint8_t, 4> color = {0, 255, 0, 255}, const float line_radius = 0.005f,
                        rerun::components::FillMode fill_mode = rerun::components::FillMode::MajorWireframe);

template <typename PointT>
std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        typename pcl::PointCloud<PointT>::Ptr& cloud, const float radius = 0.005f) {
  std::vector<rerun::Vec3D> pts;
  pts.reserve(cloud->size());
  for (const auto& p : cloud->points) {
    pts.emplace_back(p.x, p.y, p.z);
  }
  std::vector<rerun::Rgba32> colors;
  colors.reserve(cloud->size());
  for (const auto& p : cloud->points) {
    colors.emplace_back(p.r, p.g, p.b, 255);
  }
  auto point_cloud = rerun::Points3D(pts).with_colors(colors).with_radii({radius});
  std::string entity_path = entity + "/points";
  rec.log(entity_path, point_cloud);
  return entity_path;
}

template <>
std::string publishData<pcl::PointXYZI>(const rerun::RecordingStream& rec, const std::string& entity,
                                        typename pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const float radius);

template <typename PointT>
std::string publishVoxelData(const rerun::RecordingStream& rec, const std::string& entity,
                             typename pcl::PointCloud<PointT>::Ptr& cloud, const float voxel_size) {
  rerun::Boxes3D boxes;
  std::vector<rerun::Vec3D> centers;
  std::vector<rerun::Vec3D> sizes;
  centers.reserve(cloud->size());
  sizes.reserve(cloud->size());
  for (const auto& p : cloud->points) {
    rerun::Vec3D center{p.x, p.y, p.z};
    rerun::Vec3D size{voxel_size * 0.5f, voxel_size * 0.5f, voxel_size * 0.5f};
    centers.push_back(center);
    sizes.push_back(size);
  }
  std::vector<rerun::Rgba32> colors;
  colors.reserve(cloud->size());

  const float intensity_max =
      std::max_element(cloud->points.begin(), cloud->points.end(), [](const auto& a, const auto& b) {
        return a.intensity < b.intensity;
      })->intensity;

  for (const auto& p : cloud->points) {
    if (p.intensity < 0) {
      colors.emplace_back(0, 0, 255, 255);
    } else {
      double diff = intensity_max - p.intensity;
      auto c = voxblox::rainbowColorMap(diff / intensity_max);
      colors.emplace_back(c.r, c.g, c.b, 100);
    }
  }

  rerun::Boxes3D voxel_boxes = rerun::Boxes3D::from_centers_and_half_sizes(centers, sizes)
                                   .with_colors(colors)
                                   .with_fill_mode(rerun::FillMode::Solid);
  rec.log(entity, voxel_boxes);
  return entity;
}
}  // namespace dklib::perception::publisher