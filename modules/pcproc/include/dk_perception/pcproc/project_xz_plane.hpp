#pragma once
#include <pcl/common/transforms.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cassert>

namespace dklib::perception::pcproc {

/**
 * @brief Calculate the plane coefficients from an origin point and two direction vectors.
 *
 * @param origin The origin point on the plane.
 * @param v1 The first direction vector on the plane.
 * @param v2 The second direction vector on the plane.
 * @return pcl::ModelCoefficients::Ptr The coefficients of the plane.
 */
pcl::ModelCoefficients::Ptr calcPlaneFromOriginAndVectors(const Eigen::Vector3f& origin, const Eigen::Vector3f& v1,
                                                          const Eigen::Vector3f& v2) {
  Eigen::Vector3f normal = v1.cross(v2);
  normal.normalize();

  float a = normal.x();
  float b = normal.y();
  float c = normal.z();
  float d = -normal.dot(origin);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  coefficients->values.resize(4);
  coefficients->values[0] = a;
  coefficients->values[1] = b;
  coefficients->values[2] = c;
  coefficients->values[3] = d;

  return coefficients;
}

template <typename PointT>
void projectXZPlane(const typename pcl::PointCloud<PointT>::Ptr& input_cloud, const Eigen::Affine3f& transform,
                    typename pcl::PointCloud<PointT>::Ptr& output_cloud) {
  assert(input_cloud);
  assert(output_cloud);

  auto coeffs =
      calcPlaneFromOriginAndVectors(Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitZ());
  pcl::transformPointCloud(*input_cloud, *output_cloud, transform);
  pcl::ProjectInliers<pcl::PointXYZRGB> projector;
  projector.setModelType(pcl::SACMODEL_PLANE);
  projector.setInputCloud(output_cloud);
  projector.setModelCoefficients(coeffs);
  projector.filter(*output_cloud);
}
}  // namespace dklib::perception::pcproc