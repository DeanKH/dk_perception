#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include <gtsam_app/geoms.hpp>
#include <Eigen/src/Geometry/Transform.h>
#include <pcl/PolygonMesh.h>
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
#include <memory>
#include <opencv2/core.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/boxes3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/recording_stream.hpp>
#include <vector>

#include "rerun/archetypes/boxes3d.hpp"
#include "rerun/archetypes/pinhole.hpp"
#include "rerun/archetypes/points3d.hpp"
#include "rerun/archetypes/transform3d.hpp"
#include "rerun/components/view_coordinates.hpp"
#include "rerun/rotation3d.hpp"

namespace rerun {
std::string publishData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                        const Eigen::Isometry3d& transform);

std::string publishData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                        const dklib::perception::geometry::BoundingBox3D& bbox,
                        const std::array<uint8_t, 4> color = {0, 255, 0, 255}, const float line_radius = 0.005f,
                        rerun::components::FillMode fill_mode = rerun::components::FillMode::MajorWireframe);

template <typename PointT>
std::string publishData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                        typename pcl::PointCloud<PointT>::Ptr& cloud, const float radius = 0.005f) {
  const std::string entity_path = entity + "/points";
  if (!rec) {
    return entity_path;
  }

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
  rec->log(entity_path, point_cloud);
  return entity_path;
}

template <>
std::string publishData<pcl::PointXYZI>(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                        typename pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const float radius);

template <typename PointT>
std::string publishVoxelData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                             typename pcl::PointCloud<PointT>::Ptr& cloud, const float voxel_size,
                             float transparency = 0.1f) {
  if (!rec) {
    return entity;
  }
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
      colors.emplace_back(0, 0, 255, 10);
    } else {
      double diff = intensity_max - p.intensity;
      auto c = voxblox::rainbowColorMap(diff / intensity_max);
      colors.emplace_back(c.r, c.g, c.b, static_cast<uint8_t>(transparency * 255));
    }
  }

  rerun::Boxes3D voxel_boxes = rerun::Boxes3D::from_centers_and_half_sizes(centers, sizes)
                                   .with_colors(colors)
                                   .with_fill_mode(rerun::FillMode::Solid);
  rec->log(entity, voxel_boxes);
  return entity;
}

std::string publishColorImageData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                  const cv::Mat& image);
std::string publishDepthImageData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                  const cv::Mat& image, const float depth_scale);

std::string publishArrowData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                             const Eigen::Vector3d& origin, const Eigen::Vector3d& direction) {
  if (!rec) {
    return "";
  }

  std::vector<rerun::Vec3D> origins;
  origins.emplace_back(origin.x(), origin.y(), origin.z());
  std::vector<rerun::Vec3D> vectors;
  vectors.emplace_back(direction.x(), direction.y(), direction.z());

  rec->log(entity, rerun::Arrows3D::from_vectors(vectors)
                       .with_origins(origins)
                       .with_colors({rerun::Rgba32{255, 0, 0, 255}})
                       .with_radii({0.01f}));
  return entity;
}

std::string publishMeshData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr& points, const std::vector<pcl::Vertices>& polygons);

std::string publishPinholeCameraData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                     const float focal_length, const cv::Size& resolution) {
  if (!rec) {
    return entity;
  }

  rec->log(entity, rerun::Pinhole::from_focal_length_and_resolution(
                       focal_length, {static_cast<float>(resolution.width), static_cast<float>(resolution.height)})
                       .with_camera_xyz(rerun::components::ViewCoordinates::RDF));
  return entity;
}

std::string publishPinholeCameraData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                     const float focal_length, const cv::Size& resolution,
                                     const Eigen::Isometry3d& camera_pose) {
  if (!rec) {
    return entity;
  }

  publishData(rec, entity, camera_pose);
  publishPinholeCameraData(rec, entity + "/camera", focal_length, resolution);
  return entity;
}

}  // namespace rerun