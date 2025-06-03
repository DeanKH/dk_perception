// Copyright (c) 2025 deankh. All rights reserved.
#pragma once
#include <pcl/point_cloud.h>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>

namespace dklib::perception::detection::pose3d
{
/** https://arxiv.org/abs/2112.09598 で提案されている6Dビン姿勢推定手法のヒューリスティック実装
 */
class BinDetectionLukas2022Heuristic
{
public:
  template<typename PointsTraits = pcl::PointCloud<pcl::PointXYZRGB>,
    typename IteratablePointsReadOnlyAccessor = dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<PointsTraits>>
  void detect(
    const IteratablePointsReadOnlyAccessor & pointcloud)
  {
  }

private:
  const double bin_cut_interval_ = 0.05;
};

}
