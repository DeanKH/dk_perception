#pragma once
#include <pcl/PointIndices.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Geometry>
#include <dk_perception/geometry/bounding_box_3d.hpp>
#include <dk_perception/optimization/rectangle_2d_fitting.hpp>
#include <dk_perception/pcproc/project_xz_plane.hpp>
#include <dk_perception/pcproc/radial_splitter.hpp>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>
#include <stdexcept>

namespace dklib::perception::geometry {
// 隣接する点間の距離の分散によるフィルタリング
template <typename PointArrayT>
pcl::PointIndices::Ptr filterByAdjacentDistanceVariance(const PointArrayT& polygon, float sigma_threshold = 3.0f) {
  // 隣接するmin_points間の距離の分散が3σを超える点を除去
  std::vector<float> distances;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const auto& p1 = polygon[i];
    const auto& p2 = polygon[(i + 1) % polygon.size()];
    float distance = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
    distances.push_back(distance);
  }

  float mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
  float sq_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.0f);
  float stdev = std::sqrt(sq_sum / distances.size() - mean * mean);

  pcl::PointIndices::Ptr valid(new pcl::PointIndices());
  for (size_t i = 0; i < polygon.size(); ++i) {
    if (std::abs(distances[i] - mean) <= sigma_threshold * stdev) {
      valid->indices.push_back(i);
    }
  }
  return valid;
}

}  // namespace dklib::perception::geometry

namespace dklib::perception::detection::d3 {

/**
 * @brief 点群から最小値を計算する最も単純な方法.
 *
 * @tparam PointT
 * @param pc
 * @param transform_matrix
 * @return PointT
 */
template <typename PointT>
PointT calcMinimumPoint(const typename pcl::PointCloud<PointT>::Ptr& pc, const Eigen::Affine3f& transform_matrix) {
  typename pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>());
  dklib::perception::pcproc::projectXZPlane<PointT>(pc, transform_matrix, projected);
  // calc min z point
  PointT min_z_it = *std::min_element(projected->points.begin(), projected->points.end(),
                                      [](const auto& a, const auto& b) { return a.z < b.z; });

  Eigen::Affine3f inverse_transform = transform_matrix.inverse();
  min_z_it = pcl::transformPoint(min_z_it, inverse_transform);

  return min_z_it;
}

template <typename PointT>
class RadialExtremumDetector {
 public:
  RadialExtremumDetector(const pcproc::RadialSplitter<PointT>& splitter) : splitter_{splitter} {}

  std::pair<geometry::BoundingBox3D, typename pcl::PointCloud<PointT>::Ptr> execute() {
    std::vector<float> angles;
    std::vector<typename pcl::PointCloud<PointT>::Ptr> points_list;
    splitter_.detect(angles, points_list);

    assert(angles.size() == points_list.size());

    typename pcl::PointCloud<PointT>::Ptr min_points(new pcl::PointCloud<PointT>());
    min_points->reserve(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
      const auto& subset = points_list[i];
      const Eigen::AngleAxisf rotation(M_PI * -angles[i] / 180.0f, splitter_.getAxis());
      const Eigen::Affine3f transform_matrix = rotation * Eigen::Translation3f(splitter_.getCenter());

      // TODO(deankh): Implement robust extremum point search
      PointT min_z_it = calcMinimumPoint<PointT>(subset, transform_matrix);
      min_points->points.push_back(min_z_it);
    }

    // 隣接するmin_points間の距離の分散が3σを超える点を除去
    auto valid_indices =
        dklib::perception::geometry::filterByAdjacentDistanceVariance<decltype(min_points->points)>(min_points->points);

    const float plane_distance_threshold = 0.02;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    {
      pcl::SACSegmentation<pcl::PointXYZRGB> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(plane_distance_threshold);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
      seg.setInputCloud(min_points);
      seg.setIndices(valid_indices);
      seg.segment(*inliers, *coefficients);
      inliers->indices.swap(valid_indices->indices);
    }

    if (valid_indices->indices.empty()) {
      throw std::runtime_error("Could not estimate a planar model.");
    }

    // 点から平面までの距離の最大値を計算する．
    const float max_distance = [&]() {
      float max_dist = 0.0f;
      for (const auto& idx : valid_indices->indices) {
        const auto& pt = min_points->points[idx];
        float distance = std::abs(coefficients->values[0] * pt.x + coefficients->values[1] * pt.y +
                                  coefficients->values[2] * pt.z + coefficients->values[3]) /
                         std::sqrt(std::pow(coefficients->values[0], 2) + std::pow(coefficients->values[1], 2) +
                                   std::pow(coefficients->values[2], 2));
        if (distance > max_dist) {
          max_dist = distance;
        }
      }
      return max_dist;
    }();

    std::cout << "max distance: " << max_distance << std::endl;

#if 1
    // valid_indices->indicesに含まれる点列の重心を原点とする．
    const Eigen::Vector3f origin = [&]() {
      Eigen::Vector3f sum(0.0f, 0.0f, 0.0f);
      for (const auto& idx : valid_indices->indices) {
        const auto& pt = min_points->points[idx];
        sum += Eigen::Vector3f(pt.x, pt.y, pt.z);
      }
      Eigen::Vector3f center = sum / static_cast<float>(valid_indices->indices.size());
      /// 平面上に投影
      Eigen::Vector3f normal{coefficients->values[0], coefficients->values[1], coefficients->values[2]};
      normal.normalize();
      float d = coefficients->values[3];
      center = center - d * normal;
      return center;
    }();
#else
    /// 原点を平面に垂直に落とした点を計算する．
    const Eigen::Vector3f origin = [&]() {
      Eigen::Vector3f center = splitter_.getCenter();
      Eigen::Vector3f normal{coefficients->values[0], coefficients->values[1], coefficients->values[2]};
      normal.normalize();
      float d = coefficients->values[3];
      return center - d * normal;
    }();
#endif

    inline_points_ = std::make_shared<pcl::PointCloud<PointT>>();
    for (const auto& idx : valid_indices->indices) {
      pcl::PointIndices::Ptr local_valid_indices(new pcl::PointIndices());
      const auto& subset = points_list[idx];
      // subsetの点群から，平面からの距離がmax_distance以下の点群を抽出する．
      // for (const auto& pt : subset->points) {
      for (size_t i = 0; i < subset->points.size(); ++i) {
        const auto& pt = subset->points[i];
        float distance = std::abs(coefficients->values[0] * pt.x + coefficients->values[1] * pt.y +
                                  coefficients->values[2] * pt.z + coefficients->values[3]) /
                         std::sqrt(std::pow(coefficients->values[0], 2) + std::pow(coefficients->values[1], 2) +
                                   std::pow(coefficients->values[2], 2));
        if (distance <= max_distance) {
          local_valid_indices->indices.push_back(i);
          inline_points_->points.push_back(pt);
        }
      }
      if (local_valid_indices->indices.empty()) {
        std::cerr << "No valid points found in subset." << std::endl;
        continue;
      }

      // subsetの内，local_valid_indicesに含まれる点群をProjectInliersで平面に射影する．
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr subset_projected(new pcl::PointCloud<pcl::PointXYZRGB>());
      {
        pcl::ProjectInliers<pcl::PointXYZRGB> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(subset);
        proj.setIndices(local_valid_indices);
        proj.setModelCoefficients(coefficients);
        proj.filter(*subset_projected);
      }

      // subset_projectedの点群から，originに最も近い点を抽出する．
      pcl::PointXYZRGB closest_point = *std::min_element(
          subset_projected->points.begin(), subset_projected->points.end(), [&](const auto& a, const auto& b) {
            float dist_a = std::sqrt(std::pow(a.x - origin.x(), 2) + std::pow(a.y - origin.y(), 2) +
                                     std::pow(a.z - origin.z(), 2));
            float dist_b = std::sqrt(std::pow(b.x - origin.x(), 2) + std::pow(b.y - origin.y(), 2) +
                                     std::pow(b.z - origin.z(), 2));
            return dist_a < dist_b;
          });
      // update min_point[idx]
      min_points->points[idx] = closest_point;
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(min_points);
    extract.setIndices(valid_indices);
    extract.setNegative(false);
    extract.filter(*min_points);

#if 1
    // pcl::PCAで第1主成分と第2主成分を計算し，矩形の初期値とする.
    pcl::PCA<PointT> pca;
    pca.setInputCloud(min_points);
    Eigen::Vector3f plane_x = pca.getEigenVectors().col(0);
    Eigen::Vector3f plane_y = pca.getEigenVectors().col(1);
    std::cout << "pc1: " << plane_x.transpose() << std::endl;
    std::cout << "pc2: " << plane_y.transpose() << std::endl;

    // plane_xとplane_yが直交していない場合，plane_yをplane_xに直交するように修正する.
    if (std::abs(plane_x.dot(plane_y)) > 1e-3) {
      plane_y = plane_y - plane_x * plane_x.dot(plane_y);
      plane_y.normalize();
    }
    Eigen::Vector3f plane_z = plane_x.cross(plane_y);
    plane_z.normalize();

    // min_pointsの重心を計算する.
    Eigen::Vector3f centroid = pca.getMean().head(3);

    // min_pointsの点群を，原点がcentroidのplane_xとplane_yの軸からなる平面に射影する.
    Eigen::Matrix4d projection_matrix = Eigen::Matrix4d::Identity();
    projection_matrix.block<3, 1>(0, 0) = plane_x.cast<double>();
    projection_matrix.block<3, 1>(0, 1) = plane_y.cast<double>();
    projection_matrix.block<3, 1>(0, 2) = plane_z.cast<double>();
    projection_matrix.block<3, 1>(0, 3) = centroid.cast<double>();
    std::vector<Eigen::Vector2f> projected_points_2d;
    projected_points_2d.reserve(min_points->points.size());
    {
      Eigen::Matrix4f inverse_projection_matrix = projection_matrix.inverse().cast<float>();
      typename pcl::PointCloud<PointT>::Ptr projected_points(new pcl::PointCloud<PointT>());
      pcl::transformPointCloud(*min_points, *projected_points, inverse_projection_matrix);
      for (const auto& pt : projected_points->points) {
        projected_points_2d.emplace_back(pt.x, pt.y);
      }
    }

    // projected_pointsのxとyの最小値と最大値を計算する.
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const Eigen::Vector2f& pt : projected_points_2d) {
      if (pt.x() < min_x) {
        min_x = pt.x();
      }
      if (pt.x() > max_x) {
        max_x = pt.x();
      }
      if (pt.y() < min_y) {
        min_y = pt.y();
      }
      if (pt.y() > max_y) {
        max_y = pt.y();
      }
    }

    float initial_width = max_x - min_x;
    float initial_height = max_y - min_y;
    std::cout << "initial centroid: " << centroid.transpose() << std::endl;
    std::cout << "initial width: " << initial_width << ", height: " << initial_height << std::endl;

    dklib::perception::optimization::Rectangle2dFitting rect_fitter;

    gtsam::Pose2 initial_pose(0.0, 0.0, 0.0);
    gtsam::Point2 initial_size(initial_width, initial_height);

    // auto noise = gtsam::noiseModel::Isotropic::Sigma(2, 0.05);
    geometry::BoundingBox3D bbox;
    try {
      auto [optimized_pose, optimized_size] = rect_fitter.Optimize(projected_points_2d, initial_pose, initial_size);
      std::cout << "optimized pose: " << optimized_pose << std::endl;
      std::cout << "optimized size: " << optimized_size.transpose() << std::endl;  // size: width, height

      // projection_matrixで平面に投影された最適化された矩形(optimized_pose, optimized_size)を3Dに戻す.
      {
        Eigen::Vector3d optimized_pose_3d{optimized_pose.x(), optimized_pose.y(), 0.0f};
        Eigen::Vector3d bbox_center = projection_matrix.block<3, 3>(0, 0) * optimized_pose_3d + centroid.cast<double>();
        Eigen::Quaterniond rot_z(Eigen::AngleAxisd(optimized_pose.theta(), Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = projection_matrix.block<3, 1>(0, 0);
        rot_matrix.col(1) = projection_matrix.block<3, 1>(0, 1);
        rot_matrix.col(2) = projection_matrix.block<3, 1>(0, 2);
        Eigen::Quaterniond plane_orientation(rot_matrix);
        Eigen::Quaterniond bbox_orientation = plane_orientation * rot_z;

        Eigen::Vector3d size_vec{optimized_size.x(), optimized_size.y(), 0.0};
        bbox = geometry::BoundingBox3D(bbox_center, size_vec, bbox_orientation);
      }
    } catch (const std::exception& e) {
      std::cerr << "Optimization failed: " << e.what() << std::endl;
    }
#endif

    return {bbox, min_points};
  }

  typename pcl::PointCloud<PointT>::Ptr getInlinePoints() const { return inline_points_; }

 private:
  typename pcl::PointCloud<PointT>::Ptr inline_points_;

  const pcproc::RadialSplitter<PointT>& splitter_;
};

}  // namespace dklib::perception::detection::d3
