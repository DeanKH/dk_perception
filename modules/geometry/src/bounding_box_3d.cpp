#include <dk_perception/geometry/bounding_box_3d.hpp>

namespace dklib::perception::geometry {
BoundingBox3D::BoundingBox3D()
    : center(Eigen::Vector3d::Zero()), size(Eigen::Vector3d::Zero()), orientation(Eigen::Quaterniond::Identity()) {}

BoundingBox3D::BoundingBox3D(const Eigen::Vector3d& c, const Eigen::Vector3d& sz, const Eigen::Quaterniond& ori)
    : center(c), size(sz), orientation(ori) {}

Eigen::Matrix4d BoundingBox3D::getTransformation() const {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = orientation.toRotationMatrix().cast<double>();
  transform.block<3, 1>(0, 3) = center.cast<double>();
  return transform;
}

Eigen::Isometry3d BoundingBox3D::getIsometry() const {
  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = orientation.toRotationMatrix().cast<double>();
  isometry.translation() = center.cast<double>();
  return isometry;
}

BoundingBox3D BoundingBox3D::from_center_size(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                                              const Eigen::Quaterniond& orientation) {
  return BoundingBox3D(center, size, orientation);
}
}  // namespace dklib::perception::geometry