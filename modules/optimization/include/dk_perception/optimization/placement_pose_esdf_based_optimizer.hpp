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
        return true;
      }();

      // boxの上端4点がesdfグリッド内に存在するか確認する
      Eigen::Isometry3d box_iso = target_box.getIsometry();
      const bool is_box_top_inside = [&]() {
        box_iso.translation() = Eigen::Vector3d(point.x, point.y, point.z);
        std::vector<Eigen::Vector3d> box_top_points;
        box_top_points.emplace_back(target_box.size.x() * 0.5, target_box.size.y() * 0.5, target_box.size.z() * 0.5);
        box_top_points.emplace_back(-target_box.size.x() * 0.5, target_box.size.y() * 0.5, target_box.size.z() * 0.5);
        box_top_points.emplace_back(target_box.size.x() * 0.5, -target_box.size.y() * 0.5, target_box.size.z() * 0.5);
        box_top_points.emplace_back(-target_box.size.x() * 0.5, -target_box.size.y() * 0.5, target_box.size.z() * 0.5);

        bool all_points_valid = true;
        for (const auto& pt : box_top_points) {
          Eigen::Vector3d grid_point_at_sdf = box_iso * pt;
          if (grid_point_at_sdf.z() <= 0.0) {
            all_points_valid = false;
            break;
          }
        }
        return all_points_valid;
      }();

      if (is_inside && is_box_top_inside) {
        filtered_cloud->push_back(point);
      }
    }

    return filtered_cloud;
  }

  std::optional<geometry::BoundingBox3D> optimizePlacementPose(
      const geometry::BoundingBox3D& target_box, const voxblox::EsdfMap::Ptr& esdf_map,
      typename pcl::PointCloud<pcl::PointXYZI>::Ptr candidate_points = nullptr) {
    const double box_min_radius = 0.5 * std::min({target_box.size.x(), target_box.size.y(), target_box.size.z()});
    const double box_max_radius =
        0.5 * std::sqrt(target_box.size.x() * target_box.size.x() + target_box.size.y() * target_box.size.y() +
                        target_box.size.z() * target_box.size.z());

    auto placeable_candidates =
        computePlaceableCandidates(target_box, esdf_map, box_max_radius, box_max_radius + esdf_map->voxel_size());
    if (!placeable_candidates || placeable_candidates->empty()) {
      return std::nullopt;
    }

    if (candidate_points) {
      *candidate_points = *placeable_candidates;
    }

    const auto half_voxel_size = esdf_map->voxel_size();
    // placeable_candidatesをz軸(降順)->y軸(昇順)->x軸(昇順)でソートする
    std::sort(placeable_candidates->points.begin(), placeable_candidates->points.end(),
              [half_voxel_size](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
                if (std::fabs(a.z - b.z) > half_voxel_size) {
                  return a.z > b.z;
                }
                if (std::fabs(a.y - b.y) > half_voxel_size) {
                  return a.y < b.y;
                }
                return a.x < b.x;
              });

    pcl::PointXYZI best_point = placeable_candidates->points.front();
    geometry::BoundingBox3D optimized_box = target_box;
    optimized_box.center = Eigen::Vector3d(best_point.x, best_point.y, best_point.z);
    return optimized_box;
  }

  geometry::BoundingBox3D refinePlacementPose(const geometry::BoundingBox3D& initial_placement_box,
                                              const voxblox::EsdfMap::Ptr& esdf_map,
                                              const geometry::BoundingBox3D& boundary_box) {
    const Eigen::Vector3d boundary_min_far_point(-boundary_box.size.x() * 0.5, -boundary_box.size.y() * 0.5,
                                                 initial_placement_box.center.z() + initial_placement_box.size.z());

    std::vector<Eigen::Vector3d> local_box_bottom_points;
    local_box_bottom_points.emplace_back(initial_placement_box.size.x() * 0.5, initial_placement_box.size.y() * 0.5,
                                         -initial_placement_box.size.z() * 0.5);
    local_box_bottom_points.emplace_back(-initial_placement_box.size.x() * 0.5, initial_placement_box.size.y() * 0.5,
                                         -initial_placement_box.size.z() * 0.5);
    local_box_bottom_points.emplace_back(initial_placement_box.size.x() * 0.5, -initial_placement_box.size.y() * 0.5,
                                         -initial_placement_box.size.z() * 0.5);
    local_box_bottom_points.emplace_back(-initial_placement_box.size.x() * 0.5, -initial_placement_box.size.y() * 0.5,
                                         -initial_placement_box.size.z() * 0.5);

    const Eigen::Isometry3d box_iso = initial_placement_box.getIsometry();
    std::vector<Eigen::Vector3d> sdf_box_bottom_points;
    sdf_box_bottom_points.reserve(local_box_bottom_points.size());
    for (const auto& local_point : local_box_bottom_points) {
      sdf_box_bottom_points.push_back(box_iso * local_point);
    }

    // sdf_box_bottom_pointsの中で，boundary_min_far_pointに最も近い点を探す
    Eigen::Vector3d nearest_point = sdf_box_bottom_points.front();
    double nearest_distance = (sdf_box_bottom_points.front() - boundary_min_far_point).norm();
    for (const auto& pt : sdf_box_bottom_points) {
      double distance = (pt - boundary_min_far_point).norm();
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest_point = pt;
      }
    }

    double distance = 0.0;
    Eigen::Vector3d gradient;
    if (!esdf_map->getDistanceAndGradientAtPosition(nearest_point, &distance, &gradient)) {
    }

    std::cout << "refinePlacementPose: nearest_distance = " << nearest_distance << ", esdf_distance = " << distance
              << ", gradient = " << gradient.transpose() << std::endl;

    geometry::BoundingBox3D refined_box = initial_placement_box;
    refined_box.center += gradient.normalized() * distance;
    return refined_box;
  }
};

}  // namespace dklib::perception::optimization