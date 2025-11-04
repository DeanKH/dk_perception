#include "dk_perception/rerun/publish_data.hpp"

#include <limits>
#include <opencv2/opencv.hpp>

#include "rerun.hpp"
#include "rerun/archetypes/depth_image.hpp"
#include "rerun/components/fill_mode.hpp"

namespace rerun {
std::string publishData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                        const Eigen::Isometry3d& transform) {
  if (!rec) {
    return entity;
  }

  auto translation = rerun::Vec3D{transform.translation().cast<float>().x(), transform.translation().cast<float>().y(),
                                  transform.translation().cast<float>().z()};
  Eigen::Quaterniond q;
  q = transform.rotation();
  auto orientation = rerun::Quaternion::from_wxyz(q.w(), q.x(), q.y(), q.z());
  auto rotation = rerun::Rotation3D(orientation);
  auto tf = rerun::Transform3D::from_translation_rotation(translation, rotation);
  rec->log(entity, tf);

  return entity;
}

std::string publishData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                        const dklib::perception::geometry::BoundingBox3D& bbox, const std::array<uint8_t, 4> color,
                        const float line_radius, rerun::components::FillMode fill_mode) {
  const std::string entity_path = entity + "/bbox";
  if (!rec) {
    return entity_path;
  }

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
                   .with_radii({line_radius})
                   .with_fill_mode(fill_mode);
  rec->log(entity_path, boxes);
  return entity_path;
}

template <>
std::string publishData<pcl::PointXYZI>(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                        typename pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const float radius) {
  const std::string entity_path = entity + "/points";
  if (!rec) {
    return entity_path;
  }

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
  rec->log(entity_path, point_cloud);
  return entity_path;
}

std::string publishColorImageData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                  const cv::Mat& image) {
  if (!rec) {
    return entity;
  }
  assert(image.depth() == CV_8U);
  assert(image.channels() == 3);

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::vector<uint8_t> data(image.total() * image.channels());
  std::memcpy(data.data(), image.data, image.total() * image.channels() * sizeof(uint8_t));

  rec->log(entity,
           rerun::Image::from_rgb24(data, {static_cast<uint32_t>(image.cols), static_cast<uint32_t>(image.rows)}));
  return entity;
}

std::string publishDepthImageData(const std::shared_ptr<rerun::RecordingStream>& rec, const std::string& entity,
                                  const cv::Mat& image, const float depth_factor) {
  if (!rec) {
    return entity;
  }
  assert(image.depth() == CV_16U);
  assert(image.channels() == 1);

  std::vector<uint16_t> data(image.total(), std::numeric_limits<uint16_t>::max());
  std::memcpy(data.data(), image.data, image.total() * sizeof(uint16_t));

  rec->log(entity,
           rerun::DepthImage(data.data(), {static_cast<uint32_t>(image.cols), static_cast<uint32_t>(image.rows)})
               .with_meter(depth_factor)
               .with_colormap(rerun::components::Colormap::Turbo));
  return entity;
}

}  // namespace rerun