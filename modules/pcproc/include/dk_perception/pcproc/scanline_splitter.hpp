#pragma once
#include <pcl/PointIndices.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/types.h>

namespace dklib::perception::pcproc {
template <typename PointT = pcl::PointXYZ>
class ScanlineSplitter {
 public:
  enum class SplitAxis {
    kX = 0,
    kY,
    kUnknown,
  };

  std::vector<pcl::PointIndices::Ptr> splitByScanLines(const typename pcl::PointCloud<PointT>::Ptr& pointcloud,
                                                       SplitAxis axis) {
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*pointcloud, min_pt, max_pt);

    auto func = [](const typename pcl::PointCloud<PointT>::Ptr& pointcloud, const float min_value,
                   const float max_value, float scan_interval, const std::string& axis_field_name) {
      std::vector<pcl::PointIndices::Ptr> indices_vec;

      for (float scan_current = min_value; scan_current < max_value; scan_current += scan_interval) {
        float scan_next = scan_current + scan_interval;
        auto indices = splitAlongAxis(pointcloud, scan_current, scan_next, axis_field_name);
        indices_vec.push_back(indices);
      }
      return indices_vec;
    };

    std::vector<pcl::IndicesPtr> indices_vec;

    switch (axis) {
      case SplitAxis::kX:
        return func(pointcloud, min_pt[0], max_pt[0], scan_interval_, "x");
      case SplitAxis::kY:
        return func(pointcloud, min_pt[1], max_pt[1], scan_interval_, "y");
      default:
        throw std::runtime_error("Unknown axis");
    }
  }

 private:
  static pcl::PointIndices::Ptr splitAlongAxis(const typename pcl::PointCloud<PointT>::ConstPtr& pointcloud,
                                               float min_x, float max_x, const std::string& axis) {
    pcl::PassThrough<PointT> passthrough;
    passthrough.setInputCloud(pointcloud);
    passthrough.setFilterFieldName(axis);
    passthrough.setFilterLimits(min_x, max_x);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    passthrough.filter(indices->indices);
    return indices;
  }

  double scan_interval_ = 0.05;
};
}  // namespace dklib::perception::pcproc
