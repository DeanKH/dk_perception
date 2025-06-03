// Copyright (c) 2025 deankh. All rights reserved.
#pragma once
#include <cstdint>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include "dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp"

namespace dklib::perception::image_processing
{
class PointCloudToOrthoImageProjector
{
public:
  PointCloudToOrthoImageProjector(
    const Eigen::Vector3f & translation_to_plane_origin,
    const Eigen::Quaternionf & rotation_to_plane)
  : translation_to_plane_origin_{translation_to_plane_origin},
    rotation_to_plane_{rotation_to_plane}
  {

  }

  void setProjectedImageSize(std::optional<cv::Size> projected_image_size)
  {
    projected_image_size_ = projected_image_size;
  }

  template<
    typename Compare,
    typename PointsTraits = pcl::PointCloud<pcl::PointXYZRGB>,
    typename IteratablePointsReadOnlyAccessor = dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<PointsTraits>>
  cv::Mat project(
    const IteratablePointsReadOnlyAccessor & accessor, const float meter_per_pixel,
    Compare comp)
  {
    // TODO: compの型チェック
    // TODO: accessorのチェック
    // TODO: projected_image_sizeがなかったときに点群の最大値と最小値から画像サイズを決定する

    // static_assert(
    //   typeid(comp) == typeid(std::less<float>) || typeid(comp) == typeid(std::greater<float>),
    //   "Compare is not supported function, use std::less<float> or std::greater<float>");

    if (!projected_image_size_) {
      throw std::runtime_error("projected_image_size is not set");
    }
    cv::Size projected_image_size = projected_image_size_.value();
    cv::Mat projected_image = cv::Mat::zeros(
      projected_image_size,
      accessor.hasRGB() ? CV_8UC3 : CV_8UC1);
    cv::Mat projected_depth_image;
    // if Compare is std::less<float>, initialize with std::numeric_limits<float>::max()
    // if Compare is std::greater<float>, initialize with std::numeric_limits<float>::min()
    float min_depth = -std::numeric_limits<float>::max();
    float max_depth = std::numeric_limits<float>::max();
    if (typeid(comp) == typeid(std::less<float>)) {
      projected_depth_image = cv::Mat::ones(
        projected_image_size,
        CV_32FC1) * max_depth;
    } else if (typeid(comp) == typeid(std::greater<float>)) {
      projected_depth_image = cv::Mat::ones(
        projected_image_size,
        CV_32FC1) * min_depth;
    } else {
      throw std::runtime_error(
              "Compare is not supported function, use std::less<float> or std::greater<float>");
    }


    for (size_t i = 0; i < accessor.size(); i++) {
      auto point = accessor.point_at(i);
      decltype(point) point_on_plane = rotation_to_plane_ * point + translation_to_plane_origin_;
      float z = point_on_plane[2];
      int x_on_projected_image =
        static_cast<int>(point_on_plane[0] / meter_per_pixel + projected_image_size.width / 2);
      int y_on_projected_image =
        static_cast<int>(-point_on_plane[1] / meter_per_pixel + projected_image_size.height / 2);

      if (x_on_projected_image < 0 || x_on_projected_image >= projected_image_size.width ||
        y_on_projected_image < 0 || y_on_projected_image >= projected_image_size.height)
      {
        continue;
      }

      auto & current_z =
        projected_depth_image.at<float>(y_on_projected_image, x_on_projected_image);
      if (comp(z, current_z)) {
        current_z = z;
        if (accessor.hasRGB()) {
          auto color = accessor.color_at(i).value();
          projected_image.at<cv::Vec3b>(y_on_projected_image, x_on_projected_image) =
            cv::Vec3b{color[0], color[1], color[2]};
        } else if (accessor.hasMono()) {
          auto mono = accessor.mono_at(i).value();
          projected_image.at<uint8_t>(y_on_projected_image, x_on_projected_image) = mono;
        } else {
          projected_image.at<uint8_t>(
            y_on_projected_image,
            x_on_projected_image) = std::numeric_limits<uint8_t>::max();
        }
      }
    }

    return projected_image;
  }

private:
  // float meter_per_pixel_;
  Eigen::Vector3f translation_to_plane_origin_;
  Eigen::Quaternionf rotation_to_plane_;
  std::optional<cv::Size> projected_image_size_;

};
}
