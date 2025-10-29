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
    // before 65ms to process

    const Eigen::Vector3f new_y_axis = axis_.cross(ref_dir).normalized();
    // const Eigen::Vector3f new_x_axis = new_y_axis.cross(axis_).normalized();
    Eigen::Matrix3f rotation_matrix;
    rotation_matrix.col(0) = ref_dir;
    rotation_matrix.col(1) = new_y_axis;
    rotation_matrix.col(2) = axis_;

    // 法線がaxis_で新しいX軸がref_dirの平面に投影する
    {
      typename pcl::PointCloud<PointT>::Ptr projected_cloud(new pcl::PointCloud<PointT>());
      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.linear() = rotation_matrix.transpose();
      transform.translation() = -rotation_matrix.transpose() * center_;
      pcl::transformPointCloud(*input_cloud_, *projected_cloud, transform);
      // zの値を無視する

      // projected_cloudのXY成分でベクトルを作成し，X軸正方向とのなす角を0~2piで計算する.
      std::vector<float> angles_rad;
      angles_rad.reserve(projected_cloud->size());
      for (const auto& pt : projected_cloud->points) {
        float angle = -std::atan2(pt.x, pt.y);
        if (angle < 0) {
          angle += 2.0f * M_PI;
        }
        angles_rad.push_back(angle);
      }
      std::vector<size_t> sorted_indices(angles_rad.size());
      std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
      // angles_radでソートしたときのインデックスを取得する.angle_rad自体もソートする
      std::sort(sorted_indices.begin(), sorted_indices.end(),
                [&angles_rad](size_t a, size_t b) { return angles_rad[a] < angles_rad[b]; });
      std::vector<float> sorted_angles;
      sorted_angles.reserve(angles_rad.size());
      for (size_t idx : sorted_indices) {
        sorted_angles.push_back(angles_rad[idx]);
      }

      // 前のloopのupper_boundを記録しておいて高速化する
      // std::vector<float>::iterator last_upper_it = sorted_angles.begin();
      for (int i = 0; i < num_steps; ++i) {
        const float angle_start = i * angle_deg_step_;
        const float angle_end = angle_start + angle_deg_step_;
        angles.push_back(angle_start);
        // sorted_angles からangle_start~angle_endの範囲にある点のインデックスを取得する.
        // std::upper_boundを使って高速化
        pcl::PointIndices::Ptr final_indices(new pcl::PointIndices());
        auto lower_it = std::lower_bound(sorted_angles.begin(), sorted_angles.end(), angle_start * M_PI / 180.0f);
        if (lower_it == sorted_angles.end()) {
          // 範囲外
          point_indices_list[i] = final_indices;
          continue;
        }

        auto upper_it = std::upper_bound(lower_it, sorted_angles.end(), angle_end * M_PI / 180.0f);
        if (upper_it == lower_it) {
          // 範囲外
          point_indices_list[i] = final_indices;
          continue;
        }

        // iterator間の距離でfinal_indicesの要素数を確保する
        final_indices->indices.reserve(std::distance(lower_it, upper_it));
        for (auto it = lower_it; it != upper_it; ++it) {
          size_t sorted_idx = std::distance(sorted_angles.begin(), it);
          size_t original_idx = sorted_indices[sorted_idx];
          final_indices->indices.push_back(static_cast<int>(original_idx));
        }

        point_indices_list[i] = final_indices;
      }
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