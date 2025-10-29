#include "dk_perception/rerun/publish_data.hpp"

namespace dklib::perception::publisher {
std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const Eigen::Isometry3d& transform) {
  auto translation = rerun::Vec3D{transform.translation().cast<float>().x(), transform.translation().cast<float>().y(),
                                  transform.translation().cast<float>().z()};
  Eigen::Quaterniond q;
  q = transform.rotation();
  auto orientation = rerun::Quaternion::from_wxyz(q.w(), q.x(), q.y(), q.z());
  auto rotation = rerun::Rotation3D(orientation);
  auto tf = rerun::Transform3D::from_translation_rotation(translation, rotation);
  rec.log(entity, tf);

  return entity;
}

std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const dklib::perception::geometry::BoundingBox3D& bbox, const std::array<uint8_t, 4> color,
                        const float line_radius) {
  std::vector<rerun::Vec3D> centers = {
      {bbox.center.cast<float>().x(), bbox.center.cast<float>().y(), bbox.center.cast<float>().z()}};
  std::vector<rerun::Vec3D> sizes = {
      {bbox.size.cast<float>().x(), bbox.size.cast<float>().y(), bbox.size.cast<float>().z()}};

  Eigen::Quaternionf q = bbox.orientation.cast<float>();
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

template <>
std::string publishData<pcl::PointXYZI>(const rerun::RecordingStream& rec, const std::string& entity,
                                        typename pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const float radius) {
  std::vector<rerun::Vec3D> pts;
  std::vector<rerun::Rgba32> colors;
  pts.reserve(cloud->size());
  colors.reserve(cloud->size());

  const float intensity_max =
      std::max_element(cloud->points.begin(), cloud->points.end(), [](const auto& a, const auto& b) {
        return a.intensity < b.intensity;
      })->intensity;
  const float intensity_min =
      std::min_element(cloud->points.begin(), cloud->points.end(), [](const auto& a, const auto& b) {
        return a.intensity < b.intensity;
      })->intensity;
  std::cout << "Intensity min/max: " << intensity_min << " / " << intensity_max << std::endl;

  for (const auto& p : cloud->points) {
    pts.emplace_back(p.x, p.y, p.z);
    if (p.intensity > 0) {
      auto a = intensity_max - p.intensity;
      colors.emplace_back(255 * a / intensity_max, 0, 0, 255);
    } else {
      auto a = p.intensity - intensity_min;
      colors.emplace_back(0, 255 * a / std::fabs(intensity_min), 0, 255);
    }
  }

  auto point_cloud = rerun::Points3D(pts).with_colors(colors).with_radii({radius});
  std::string entity_path = entity + "/points";
  rec.log(entity_path, point_cloud);
  return entity_path;
}

}  // namespace dklib::perception::publisher