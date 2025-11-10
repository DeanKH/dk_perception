#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace dklib::perception::geometry {
struct BoundingBox3D {
  Eigen::Vector3d center;          // 中心点
  Eigen::Vector3d size;            // 各軸方向のサイズ (幅, 高さ, 奥行き)
  Eigen::Quaterniond orientation;  // 回転 (クォータニオン)

  BoundingBox3D();

  BoundingBox3D(const Eigen::Vector3d& c, const Eigen::Vector3d& sz, const Eigen::Quaterniond& ori);

  Eigen::Matrix4d getTransformation() const;
  Eigen::Isometry3d getIsometry() const;
  void setTransformation(const Eigen::Matrix4d& transform);
  void setIsometry(const Eigen::Isometry3d& isometry);

  static BoundingBox3D from_center_size(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                                        const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity());
};
}  // namespace dklib::perception::geometry