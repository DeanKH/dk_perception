#pragma once
#include <pcl/PointIndices.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/types.h>

namespace dklib::perception::pcproc {

template <typename PointT>
class RadialSplitter {
 public:
  void setInputCloud(const typename pcl::PointCloud<PointT>::Ptr& cloud) { input_cloud_ = cloud; }
  void setCenter(const Eigen::Vector3f& center) { center_ = center; }
  void setAxis(const Eigen::Vector3f& axis) { axis_ = axis.normalized(); }
  void setMinDistance(float minDistance) { min_distance_ = minDistance; }
  void setAngleStep(float angleStep) { angle_deg_step_ = angleStep; }
  void setWidth(float width) { width_ = width; }

  Eigen::Vector3f getCenter() const { return center_; }
  Eigen::Vector3f getAxis() const { return axis_; }

  void detect(std::vector<float>& angles, std::vector<pcl::PointIndices::Ptr>& point_indices_list) const {
    if (!input_cloud_ || input_cloud_->empty()) {
      std::cerr << "Input cloud is not set or empty." << std::endl;
      return;
    }

    // Precompute angles and point indices
    angles.clear();
    point_indices_list.clear();
    int num_steps = static_cast<int>(360.0f / angle_deg_step_);
    angles.reserve(num_steps);
    point_indices_list.resize(num_steps);

    Eigen::Vector3f ref_dir = axis_.unitOrthogonal();  // Reference direction on the plane orthogonal to axis_

    for (int i = 0; i < num_steps; ++i) {
      float angle = i * angle_deg_step_;
      angles.push_back(angle);

      // ref_dirをaxis_を軸にangle度回転させる回転行列を作成
      Eigen::AngleAxisf rotation(-M_PI * angle / 180.0f, axis_);
      Eigen::Quaternionf q(rotation);

      // TODO(deankh): Too slowly
      typename pcl::PointCloud<PointT>::Ptr rotated_cloud(new pcl::PointCloud<PointT>());
      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.rotate(q);
      transform.translate(center_);
      pcl::transformPointCloud(*input_cloud_, *rotated_cloud, transform);

      pcl::PassThrough<PointT> pass_y;
      pass_y.setInputCloud(rotated_cloud);
      pass_y.setFilterFieldName("y");
      pass_y.setFilterLimits(-width_ / 2.0f, width_ / 2.0f);

      pcl::PointIndices::Ptr y_indices(new pcl::PointIndices());
      pass_y.filter(y_indices->indices);

      pcl::PassThrough<PointT> pass_x;
      pass_x.setInputCloud(rotated_cloud);
      pass_x.setIndices(y_indices);
      pass_x.setFilterFieldName("x");
      pass_x.setFilterLimits(min_distance_, std::numeric_limits<float>::max());

      pcl::PointIndices::Ptr final_indices(new pcl::PointIndices());
      pass_x.filter(final_indices->indices);
      point_indices_list[i] = final_indices;
    }
  }

  void detect(std::vector<float>& angles, std::vector<typename pcl::PointCloud<PointT>::Ptr>& points_list) const {
    std::vector<pcl::PointIndices::Ptr> point_indices_list;
    detect(angles, point_indices_list);

    points_list.clear();
    points_list.resize(point_indices_list.size());
    for (size_t i = 0; i < point_indices_list.size(); ++i) {
      points_list[i] = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
      if (point_indices_list[i]->indices.empty()) {
        continue;
      }

      points_list[i]->reserve(point_indices_list[i]->indices.size());
      for (int idx : point_indices_list[i]->indices) {
        points_list[i]->push_back(input_cloud_->at(idx));
      }
    }
  }

 private:
  typename pcl::PointCloud<PointT>::Ptr input_cloud_;
  Eigen::Vector3f center_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f axis_ = Eigen::Vector3f::UnitZ();
  float min_distance_ = 0.01f;
  float angle_deg_step_ = 1.0f;  // deg
  float width_ = 0.05f;
};

}  // namespace dklib::perception::pcproc