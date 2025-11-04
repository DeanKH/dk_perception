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

    // boxの底面をvoxel_size間隔の内部のグリッドを作成する
    // const double voxel_size = esdf_map->voxel_size();

    // std::optional<geometry::BoundingBox3D> best_candidate = std::nullopt;
    // for (const auto& point_candidate : placeable_candidates->points) {
    //   std::cout << "Placeable candidate at: " << point_candidate.x << ", " << point_candidate.y << ", "
    //             << point_candidate.z << ", distance: " << point_candidate.intensity << std::endl;
    //   geometry::BoundingBox3D optimized_candidate = target_box;
    //   optimized_candidate.center = Eigen::Vector3d(point_candidate.x, point_candidate.y, point_candidate.z);

    //   const Eigen::Isometry3d box_iso = optimized_candidate.getIsometry();
    //   bool has_invalid_grid = false;
    //   for (double dx = -optimized_candidate.size.x() * 0.5; dx < optimized_candidate.size.x() * 0.5; dx +=
    //   voxel_size) {
    //     for (double dy = -optimized_candidate.size.y() * 0.5; dy < optimized_candidate.size.y() * 0.5;
    //          dy += voxel_size) {
    //       const double dz = -optimized_candidate.size.z() * 0.5;
    //       Eigen::Vector3d grid_local_point(dx, dy, dz);
    //       // sdf座標系に変換する
    //       Eigen::Vector3d grid_point_at_sdf = box_iso * grid_local_point;
    //       // ESDFマップから距離と勾配を取得
    //       double distance = 0.0;
    //       Eigen::Vector3d gradient;
    //       if (!esdf_map->getDistanceAndGradientAtPosition(grid_point_at_sdf, &distance, &gradient)) {
    //         has_invalid_grid = true;
    //         break;
    //       }

    //       // 距離が負の場合は衝突している
    //       if (distance < half_voxel_size) {
    //         has_invalid_grid = true;
    //         break;
    //       }
    //     }

    //     if (has_invalid_grid) {
    //       break;
    //     }
    //   }
    //   if (has_invalid_grid) {
    //     continue;
    //   }

    //   best_candidate = optimized_candidate;
    // }

    // return best_candidate;
  }
};

}  // namespace dklib::perception::optimization