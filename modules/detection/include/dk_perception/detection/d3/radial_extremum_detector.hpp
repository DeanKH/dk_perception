#pragma once
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>

#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>

namespace dklib::perception::detection::d3 {

template <typename PointT>
class RadialExtremumDetector {
 public:
  RadialExtremumDetector() {}

  void setInputCloud(const typename pcl::PointCloud<PointT>::Ptr& cloud) { input_cloud_ = cloud; }
  void setCenter(const Eigen::Vector3f& center) { center_ = center; }
  void setAxis(const Eigen::Vector3f& axis) { axis_ = axis.normalized(); }
  void setMinDistance(float minDistance) { min_distance_ = minDistance; }
  void setAngleStep(float angleStep) { angle_step_ = angleStep; }
  void setWidth(float width) { width_ = width; }

  void detect(std::vector<float>& angles, std::vector<pcl::PointIndices>& point_indices) {
    angles.clear();
    point_indices.clear();

    if (!input_cloud_ || input_cloud_->empty()) {
      std::cerr << "Input cloud is not set or empty." << std::endl;
      return;
    }

    // Precompute angles and point indices
    angles.clear();
    point_indices.clear();
    int num_steps = static_cast<int>(360.0f / angle_step_);
    std::cout << "Number of steps: " << num_steps << std::endl;
    angles.reserve(num_steps);
    point_indices.resize(num_steps);

    Eigen::Vector3f ref_dir = axis_.unitOrthogonal();  // Reference direction on the plane orthogonal to axis_

    for (int i = 0; i < num_steps; ++i) {
      float angle = i * angle_step_;
      std::cout << "step " << i << ": angle = " << angle << std::endl;
      angles.push_back(angle);
      continue;

      // Compute the direction vector for this angle
      Eigen::AngleAxisf rotation(M_PI * angle / 180.0f, axis_);
      Eigen::Vector3f dir = rotation * ref_dir;

      // Define a plane perpendicular to the direction vector at the center
      Eigen::Vector3f plane_normal = dir;
      Eigen::Vector3f plane_point = center_;

      // Collect points within the width around the ray defined by center_ and dir
      pcl::PointIndices indices;
      for (size_t idx = 0; idx < input_cloud_->size(); ++idx) {
        const PointT& pt = input_cloud_->at(idx);
        Eigen::Vector3f point(pt.x, pt.y, pt.z);
        Eigen::Vector3f vec_to_point = point - center_;

        // Project vec_to_point onto the plane normal
        float distance_to_plane = vec_to_point.dot(plane_normal);
        Eigen::Vector3f projected_point = point - distance_to_plane * plane_normal;

        // Check if the projected point is within the width
        if (std::abs(distance_to_plane) <= width_ / 2.0f) {
          float distance_from_center = (projected_point - center_).norm();
          if (distance_from_center >= min_distance_) {
            indices.indices.push_back(static_cast<int>(idx));
          }
        }
      }
      point_indices[i] = indices;
    }

    // Find extremum points in each sector
    for (size_t i = 0; i < angles.size(); ++i) {
      const pcl::PointIndices& indices = point_indices[i];
      if (indices.indices.empty()) {
        continue;
      }
    }
  }

 private:
  typename pcl::PointCloud<PointT>::Ptr input_cloud_;
  Eigen::Vector3f center_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f axis_ = Eigen::Vector3f::UnitZ();
  float min_distance_ = 0.01f;
  float angle_step_ = 1.0f;  // deg
  float width_ = 0.05f;

  // std::vector<float> angles_;
  // std::vector<pcl::PointIndices> point_indices_;
};

}  // namespace dklib::perception::detection::d3