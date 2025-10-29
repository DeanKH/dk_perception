#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <voxblox/core/esdf_map.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>

#include <dk_perception/geometry/bounding_box_3d.hpp>
#include <dk_perception/reconstruction/voxblox_pcl_conversion.hpp>
#include <limits>
#include <optional>

namespace dklib::perception::optimization {
class PlacementPoseEsdfBasedOptimizer {
 public:
  pcl::PointCloud<pcl::PointXYZI>::Ptr computePlaceableCandidates(const geometry::BoundingBox3D& target_box,
                                                                  const voxblox::EsdfMap::Ptr& esdf_map,
                                                                  const double min_distance_from_center,
                                                                  const double max_distance_from_center) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());

    voxblox::createDistancePointcloudFromEsdfLayer(esdf_map->getEsdfLayer(), cloud.get());

    // filter by box
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    filtered_cloud->reserve(cloud->size());
    const double voxel_size = esdf_map->voxel_size();
    const double voxel_half_size = voxel_size * 0.5;
    for (const auto& point : cloud->points) {
      const bool is_inside = [&]() {
        Eigen::Vector3d p(point.x, point.y, point.z);
        if (point.intensity < min_distance_from_center || point.intensity > max_distance_from_center) {
          return false;
        }
        filtered_cloud->points.push_back(point);
        return true;
      }();
      if (is_inside) {
        filtered_cloud->push_back(point);
      }
    }

    return filtered_cloud;
  }

  std::optional<geometry::BoundingBox3D> optimizePlacementPose(const geometry::BoundingBox3D& target_box,
                                                               const voxblox::EsdfMap::Ptr& esdf_map) {
    const double box_min_radius = 0.5 * std::min({target_box.size.x(), target_box.size.y(), target_box.size.z()});
    const double box_max_radius =
        0.5 * std::sqrt(target_box.size.x() * target_box.size.x() + target_box.size.y() * target_box.size.y() +
                        target_box.size.z() * target_box.size.z());

    auto placeable_candidates =
        computePlaceableCandidates(target_box, esdf_map, box_max_radius, box_max_radius + esdf_map->voxel_size());
    if (!placeable_candidates || placeable_candidates->empty()) {
      return std::nullopt;
    }

    // placeable_candidatesをz軸(降順)->y軸(昇順)->x軸(昇順)でソートする
    std::sort(placeable_candidates->points.begin(), placeable_candidates->points.end(),
              [](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
                if (std::fabs(a.z - b.z) > std::numeric_limits<double>::epsilon()) {
                  return a.z > b.z;
                }
                if (std::fabs(a.y - b.y) > std::numeric_limits<double>::epsilon()) {
                  return a.y < b.y;
                }
                return a.x < b.x;
              });
    const pcl::PointXYZI& best_point = placeable_candidates->points.front();

    geometry::BoundingBox3D optimized_box = target_box;
    optimized_box.center = Eigen::Vector3d(best_point.x, best_point.y, best_point.z);
    std::cout << "Optimized placement pose at: " << optimized_box.center.transpose()
              << ", distance: " << best_point.intensity << std::endl;

    // esdf_map->batchGetDistanceAndGradientAtPosition(EigenDRef<const Eigen::Matrix<double, 3, Eigen::Dynamic>>
    // &positions, Eigen::Ref<Eigen::VectorXd> distances, EigenDRef<Eigen::Matrix<double, 3, Eigen::Dynamic>>
    // &gradients, Eigen::Ref<Eigen::VectorXi> observed)
    return optimized_box;
  }
};

}  // namespace dklib::perception::optimization