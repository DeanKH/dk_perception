#include <dk_perception/geometry/bounding_box_3d.hpp>

namespace dklib::perception::geometry {
BoundingBox3D::BoundingBox3D()
    : center(Eigen::Vector3d::Zero()), size(Eigen::Vector3d::Zero()), orientation(Eigen::Quaterniond::Identity()) {}

BoundingBox3D::BoundingBox3D(const Eigen::Vector3d& c, const Eigen::Vector3d& sz, const Eigen::Quaterniond& ori)
    : center(c), size(sz), orientation(ori) {}

BoundingBox3D BoundingBox3D::from_center_size(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                                              const Eigen::Quaterniond& orientation) {
  return BoundingBox3D(center, size, orientation);
}
}  // namespace dklib::perception::geometry