// Copyright (c) 2025 deankh. All rights reserved.
#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "rgbd_type.hpp"

namespace dklib::perception::type::pointcloud
{

template<typename PointCloudType>
struct PointCloudTraits;

template<>
struct PointCloudTraits<pcl::PointCloud<pcl::PointXYZ>>
{
  using PointType = pcl::PointXYZ;
  using ColorType = std::array<uint8_t, 3>;

  static size_t size(const pcl::PointCloud<PointType> & point_cloud)
  {
    return point_cloud.size();
  }

  static Eigen::Vector3f point_at(
    const pcl::PointCloud<PointType> & point_cloud, size_t index)
  {
    return point_cloud.at(index).getVector3fMap();
  }

  static std::optional<ColorType> color_at(
    const pcl::PointCloud<PointType> & point_cloud,
    size_t index)
  {
    return std::nullopt;
  }

  static std::optional<uint8_t> mono_at(
    const pcl::PointCloud<PointType> & point_cloud,
    size_t index)
  {
    return std::nullopt;
  }

  static bool hasRGB(const pcl::PointCloud<PointType> & point_cloud)
  {
    return false;
  }

  static bool hasMono(const pcl::PointCloud<PointType> & point_cloud)
  {
    return false;
  }
};

template<>
struct PointCloudTraits<pcl::PointCloud<pcl::PointXYZRGB>>
{
  using PointType = pcl::PointXYZRGB;
  using ColorType = std::array<uint8_t, 3>;

  static size_t size(const pcl::PointCloud<PointType> & point_cloud)
  {
    return point_cloud.size();
  }

  static Eigen::Vector3f point_at(
    const pcl::PointCloud<PointType> & point_cloud, size_t index)
  {
    return point_cloud.at(index).getVector3fMap();
  }

  static std::optional<ColorType> color_at(
    const pcl::PointCloud<PointType> & point_cloud,
    size_t index)
  {
    pcl::Vector3cMapConst c = point_cloud.at(index).getBGRVector3cMap();
    return std::array{c(0), c(1), c(2)};
  }

  static std::optional<uint8_t> mono_at(
    const pcl::PointCloud<PointType> & point_cloud,
    size_t index)
  {
    auto color = point_cloud.at(index).getBGRVector3cMap();
    return 0.299 * color(2) + 0.587 * color(1) + 0.114 * color(0);
  }

  static bool hasRGB(const pcl::PointCloud<PointType> & point_cloud)
  {
    return true;
  }

  static bool hasMono(const pcl::PointCloud<PointType> & point_cloud)
  {
    return true;
  }
};

// sample type
struct PointSet
{
  std::vector<Eigen::Vector3f> points;
  std::vector<std::array<uint8_t, 3>> colors;
};

template<>
struct PointCloudTraits<PointSet>
{
  using PointType = Eigen::Vector3f;
  using ColorType = std::array<uint8_t, 3>;

  static size_t size(const PointSet & point_set)
  {
    return point_set.points.size();
  }

  static Eigen::Vector3f point_at(const PointSet & point_set, size_t index)
  {
    return Eigen::Map<const Eigen::Vector3f>(point_set.points[index].data());
  }

  static std::optional<ColorType> color_at(const PointSet & point_set, size_t index)
  {
    if (index < point_set.colors.size()) {
      return point_set.colors[index];
    }
    return std::nullopt;
  }

  static std::optional<uint8_t> mono_at(const PointSet & point_set, size_t index)
  {
    if (index < point_set.colors.size()) {
      auto color = point_set.colors[index];
      return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0];
    }
    return std::nullopt;
  }

  static bool hasRGB(const PointSet & point_set)
  {
    return true;
  }

  static bool hasMono(const PointSet & point_set)
  {
    return true;
  }
};

template<>
struct PointCloudTraits<DepthImageSet>
{
  using PointType = Eigen::Vector3f;
  using ColorType = std::array<uint8_t, 3>;

  static size_t size(const DepthImageSet & rgbd_image_set)
  {
    assert(rgbd_image_set.color_image().rows == rgbd_image_set.depth_image().rows);
    assert(rgbd_image_set.color_image().cols == rgbd_image_set.depth_image().cols);

    return rgbd_image_set.color_image().rows * rgbd_image_set.color_image().cols;
  }

  static Eigen::Vector3f point_at(
    const DepthImageSet & rgbd_image_set,
    size_t index)
  {
    size_t row = index / rgbd_image_set.color_image().cols;
    size_t col = index % rgbd_image_set.color_image().cols;
    auto depth = rgbd_image_set.depth_image().at<uint16_t>(row, col);
    if (depth == 0) {
      // std::cout << "depth is nan" << std::endl;
      float nan = std::numeric_limits<float>::quiet_NaN();
      PointType point = PointType(nan, nan, nan);
      return Eigen::Map<const PointType>(point.data());
    }

    float z = depth * rgbd_image_set.depth_factor();
    Eigen::Vector3f point;
    point << (col - rgbd_image_set.intrinsic()(0, 2)) * z / rgbd_image_set.intrinsic()(0, 0),
      (row - rgbd_image_set.intrinsic()(1, 2)) * z / rgbd_image_set.intrinsic()(1, 1),
      z;
    return point;
  }

  static std::optional<ColorType> color_at(
    const DepthImageSet & rgbd_image_set,
    size_t index)
  {
    size_t row = index / rgbd_image_set.color_image().cols;
    size_t col = index % rgbd_image_set.color_image().cols;
    cv::Vec3b color = rgbd_image_set.color_image().at<cv::Vec3b>(row, col);
    return std::array{color[0], color[1], color[2]};
  }

  static std::optional<uint8_t> mono_at(
    const DepthImageSet & rgbd_image_set,
    size_t index)
  {
    size_t row = index / rgbd_image_set.color_image().cols;
    size_t col = index % rgbd_image_set.color_image().cols;
    cv::Vec3b color = rgbd_image_set.color_image().at<cv::Vec3b>(row, col);
    return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0];
  }

  static bool hasRGB(const DepthImageSet & rgbd_image_set)
  {
    return rgbd_image_set.color_image().type() == CV_8UC3;
  }

  static bool hasMono(const DepthImageSet & rgbd_image_set)
  {
    return rgbd_image_set.color_image().type() == CV_8UC1;
  }
};
}
