// Copyright (c) 2025 deankh. All rights reserved.
#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include "point_traits.hpp"

namespace dklib::perception::type::pointcloud
{
template<typename PointCloudType>
class IteratableColorizedPointCloudReadOnlyAccessor
{
public:
  IteratableColorizedPointCloudReadOnlyAccessor(
    const PointCloudType & point_cloud)
  : point_cloud_(point_cloud)
  {
  }

  size_t size() const
  {
    return PointCloudTraits<PointCloudType>::size(point_cloud_);
  }

  Eigen::Vector3f point_at(size_t index) const
  {
    return PointCloudTraits<PointCloudType>::point_at(point_cloud_, index);
  }

  std::optional<typename PointCloudTraits<PointCloudType>::ColorType> color_at(size_t index) const
  {
    return PointCloudTraits<PointCloudType>::color_at(point_cloud_, index);
  }

  std::optional<uint8_t> mono_at(size_t index) const
  {
    return PointCloudTraits<PointCloudType>::mono_at(point_cloud_, index);
  }

  bool hasRGB() const
  {
    return PointCloudTraits<PointCloudType>::hasRGB(point_cloud_);
  }

  bool hasMono() const
  {
    return PointCloudTraits<PointCloudType>::hasMono(point_cloud_);
  }

private:
  const PointCloudType & point_cloud_;
};

template<typename PointCloudType>
std::ostream & operator<<(
  std::ostream & os,
  const dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<PointCloudType> & accessor)
{
  os << "point_size: " << accessor.size() << "\n";
  os << "point_at: " << accessor.point_at(0).transpose() << "\n";
  if (accessor.hasRGB()) {
    os << "color_at: " << accessor.color_at(0).value().transpose() << "\n";
  } else {
    os << "color_at: " << "no color" << "\n";
  }
  if (accessor.hasMono()) {
    os << "mono_at: " << static_cast<int>(accessor.mono_at(0).value()) << "\n";
  } else {
    os << "mono_at : " << "no mono ";
  }
  return os;
}
}
