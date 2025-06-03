// Copyright (c) 2025 deankh. All rights reserved.
#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Core>

namespace dklib::perception::type::pointcloud
{
class DepthImageSet
{
public:
  DepthImageSet(
    const cv::Mat & color_image,
    const cv::Mat & depth_image,
    const Eigen::Matrix3f & intrinsic,
    const float depth_factor)
  : color_image_(color_image), depth_image_(depth_image), intrinsic_(intrinsic), depth_factor_(
      depth_factor)
  {
  }

  const cv::Mat & color_image() const
  {
    return color_image_;
  }

  const cv::Mat & depth_image() const
  {
    return depth_image_;
  }

  const Eigen::Matrix3f & intrinsic() const
  {
    return intrinsic_;
  }

  const float & depth_factor() const
  {
    return depth_factor_;
  }

private:
  const cv::Mat color_image_; // not reference, but shallow copy
  const cv::Mat depth_image_;  // not reference, but shallow copy
  const Eigen::Matrix3f intrinsic_;
  float depth_factor_;
};
}
