#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include <gtsam_app/geoms.hpp>
#include <dk_perception/geometry.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/boxes3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/recording_stream.hpp>
#include <vector>

namespace dklib::perception::visualization {
/**
 * Publish a 2D bounding box to Rerun.
 * 高さ0, Yaw軸以外の回転なしで2Dの矩形をRerunに出力します．
 * 矩形は中心の点と矩形のポリゴンを出力します．
 */
std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const geometry::BoundingBox2D& bbox, const std::array<uint8_t, 4> color = {0, 255, 0, 255},
                        const float line_radius = 0.02f) {
  std::vector<rerun::Vec3D> centers = {{bbox.center.x(), bbox.center.y(), 0.0f}};
  std::vector<rerun::Vec3D> sizes = {{bbox.size.x(), bbox.size.y(), 0.00f}};

  Eigen::Quaternionf q;
  q = Eigen::AngleAxisf(bbox.rotation, Eigen::Vector3f::UnitZ());
  std::vector<rerun::Quaternion> orientations = {rerun::Quaternion::from_wxyz(q.w(), q.x(), q.y(), q.z())};

  std::vector<rerun::Rgba32> colors = {rerun::Rgba32{color[0], color[1], color[2], color[3]}};
  auto boxes = rerun::Boxes3D::from_centers_and_sizes(centers, sizes)
                   .with_quaternions(orientations)
                   .with_colors(colors)
                   .with_radii({line_radius});
  std::string entity_path = entity + "/bbox";
  rec.log(entity_path, boxes);
  return entity_path;
}

std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const std::vector<Eigen::Vector2f>& points,
                        const std::array<uint8_t, 4>& color = {0, 0, 255, 255}, const float radius = 0.02f) {
  std::vector<rerun::Vec3D> pts;
  pts.reserve(points.size());
  for (const auto& p : points) {
    pts.emplace_back(p.x(), p.y(), 0.0f);
  }
  auto point_cloud =
      rerun::Points3D(pts)
          .with_colors(std::vector<rerun::Rgba32>(pts.size(), rerun::Rgba32{color[0], color[1], color[2], color[3]}))
          .with_radii({radius});
  std::string entity_path = entity + "/points";
  rec.log(entity_path, point_cloud);
  return entity_path;
}
}  // namespace dklib::perception::visualization