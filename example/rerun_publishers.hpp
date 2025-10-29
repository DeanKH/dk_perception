#pragma once
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

#include "rerun/archetypes/boxes3d.hpp"
#include "rerun/archetypes/points3d.hpp"
#include "rerun/archetypes/transform3d.hpp"
#include "rerun/rotation3d.hpp"

std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const Eigen::Isometry3d& transform) {
  auto translation = rerun::Vec3D{transform.translation().cast<float>().x(), transform.translation().cast<float>().y(),
                                  transform.translation().cast<float>().z()};
  Eigen::Quaterniond q;
  q = transform.rotation();
  auto orientation = rerun::Quaternion::from_wxyz(q.w(), q.x(), q.y(), q.z());
  auto rotation = rerun::Rotation3D(orientation);
  auto tf = rerun::Transform3D::from_translation_rotation(translation, rotation);
  rec.log(entity, tf);

  return entity;
}

std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const dklib::perception::geometry::BoundingBox3D& bbox,
                        const std::array<uint8_t, 4> color = {0, 255, 0, 255}, const float line_radius = 0.005f) {
  std::vector<rerun::Vec3D> centers = {
      {bbox.center.cast<float>().x(), bbox.center.cast<float>().y(), bbox.center.cast<float>().z()}};
  std::vector<rerun::Vec3D> sizes = {
      {bbox.size.cast<float>().x(), bbox.size.cast<float>().y(), bbox.size.cast<float>().z()}};

  Eigen::Quaternionf q = bbox.orientation.cast<float>();
  std::vector<rerun::Quaternion> orientations = {rerun::Quaternion::from_wxyz(q.w(), q.x(), q.y(), q.z())};

  std::vector<rerun::Rgba32> colors = {rerun::Rgba32{color[0], color[1], color[2], color[3]}};
  auto boxes = rerun::Boxes3D::from_centers_and_sizes(centers, sizes)
                   .with_quaternions(orientations)
                   .with_colors(colors)
                   .with_radii({line_radius});
  std::string entity_path = entity + "/bbox";
  rec.log(entity_path, boxes);
  return entity_path;
}

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
                                        typename pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const float radius) {
  std::vector<rerun::Vec3D> pts;
  std::vector<rerun::Rgba32> colors;
  pts.reserve(cloud->size());
  colors.reserve(cloud->size());

  const float intensity_max =
      std::max_element(cloud->points.begin(), cloud->points.end(), [](const auto& a, const auto& b) {
        return a.intensity < b.intensity;
      })->intensity;
  const float intensity_min =
      std::min_element(cloud->points.begin(), cloud->points.end(), [](const auto& a, const auto& b) {
        return a.intensity < b.intensity;
      })->intensity;
  std::cout << "Intensity min/max: " << intensity_min << " / " << intensity_max << std::endl;

  for (const auto& p : cloud->points) {
    pts.emplace_back(p.x, p.y, p.z);
    if (p.intensity > 0) {
      auto a = intensity_max - p.intensity;
      colors.emplace_back(255 * a / intensity_max, 0, 0, 255);
    } else {
      auto a = p.intensity - intensity_min;
      colors.emplace_back(0, 255 * a / std::fabs(intensity_min), 0, 255);
    }
  }

  auto point_cloud = rerun::Points3D(pts).with_colors(colors).with_radii({radius});
  std::string entity_path = entity + "/points";
  rec.log(entity_path, point_cloud);
  return entity_path;
}

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

inline bool visualizeOccupiedTsdfVoxels(const voxblox::TsdfVoxel& voxel, const voxblox::Point& /*coord*/,
                                        const float min_distance = 0.0) {
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight && voxel.distance <= min_distance) {
    return true;
  }
  return false;
}

namespace voxblox {
template <typename VoxelType>
using ShouldVisualizeVoxelFunctionType = std::function<bool(const VoxelType& voxel, const Point& coord)>;  // NOLINT;

template <typename VoxelType>
void createOccupancyBlocksFromLayer(const Layer<VoxelType>& layer,
                                    const ShouldVisualizeVoxelFunctionType<VoxelType>& vis_function,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& occupied_points) {
  assert(occupied_points);

  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;
  FloatingPoint voxel_size = layer.voxel_size();

  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);
  for (const BlockIndex& index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord)) {
        pcl::PointXYZRGB cube_center;
        cube_center.x = coord.x();
        cube_center.y = coord.y();
        cube_center.z = coord.z();
        auto color = rainbowColorMap((coord.z() + 2.5) / 5.0);
        cube_center.r = color.r;
        cube_center.g = color.g;
        cube_center.b = color.b;
        occupied_points->points.push_back(cube_center);
      }
    }
  }
}

inline void createOccupancyBlocksFromTsdfLayer(const voxblox::Layer<voxblox::TsdfVoxel>& layer,
                                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr& occupied_points) {
  assert(occupied_points);

  createOccupancyBlocksFromLayer<voxblox::TsdfVoxel>(
      layer, std::bind(visualizeOccupiedTsdfVoxels, std::placeholders::_1, std::placeholders::_2, layer.voxel_size()),
      occupied_points);
}
#if 0
inline bool visualizeDistanceIntensityTsdfVoxels(const TsdfVoxel& voxel, const Point& /*coord*/, double* intensity) {
  assert(intensity);
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight) {
    *intensity = voxel.distance;
    return true;
  }
  return false;
}

/// This is a fancy alias to be able to template functions.
template <typename VoxelType>
using ShouldVisualizeVoxelColorFunctionType =
    std::function<bool(const VoxelType& voxel, const Point& coord, Color* color)>;

/// Template function to visualize a colored pointcloud.
template <typename VoxelType>
void createColorPointcloudFromLayer(const Layer<VoxelType>& layer,
                                    const ShouldVisualizeVoxelColorFunctionType<VoxelType>& vis_function,
                                    pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
  assert(pointcloud);
  pointcloud->clear();
  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  Color color;
  // Iterate over all blocks.
  for (const BlockIndex& index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord, &color)) {
        pcl::PointXYZRGB point;
        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();
        point.r = color.r;
        point.g = color.g;
        point.b = color.b;
        pointcloud->push_back(point);
      }
    }
  }
}

/**
 * For intensities values, such as distances, which are mapped to a color only
 * by the subscriber.
 */
template <typename VoxelType>
using ShouldVisualizeVoxelIntensityFunctionType =
    std::function<bool(const VoxelType& voxel, const Point& coord, double* intensity)>;

/// Template function to visualize an intensity pointcloud.
template <typename VoxelType>
void createColorPointcloudFromLayer(const Layer<VoxelType>& layer,
                                    const ShouldVisualizeVoxelIntensityFunctionType<VoxelType>& vis_function,
                                    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  assert(pointcloud);
  pointcloud->clear();
  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  double intensity = 0.0;
  // Iterate over all blocks.
  for (const BlockIndex& index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord, &intensity)) {
        pcl::PointXYZI point;
        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();
        point.intensity = intensity;
        pointcloud->push_back(point);
      }
    }
  }
}

/**
 * Create a pointcloud based on all the TSDF voxels.
 * The intensity is determined based on the distance to the surface.
 */
inline void createDistancePointcloudFromTsdfLayer(const Layer<TsdfVoxel>& layer,
                                                  pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  assert(pointcloud);
  createColorPointcloudFromLayer<TsdfVoxel>(layer, &visualizeDistanceIntensityTsdfVoxels, pointcloud);
}
#endif

}  // namespace voxblox
