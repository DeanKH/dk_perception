#include <gperftools/profiler.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/ply_io.h>

#include <chrono>
#include <dk_perception/detection/d3/radial_extremum_detector.hpp>
#include <filesystem>
#include <iostream>
#include <rerun.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/recording_stream.hpp>

std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        const dklib::perception::geometry::BoundingBox3D& bbox,
                        const std::array<uint8_t, 4> color = {0, 255, 0, 255}, const float line_radius = 0.005f) {
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

template <typename PointT>
std::string publishData(const rerun::RecordingStream& rec, const std::string& entity,
                        typename pcl::PointCloud<PointT>::Ptr& cloud, const float radius = 0.005f) {
  std::vector<rerun::Vec3D> pts;
  pts.reserve(cloud->size());
  for (const auto& p : cloud->points) {
    pts.emplace_back(p.x, p.y, p.z);
  }
  std::vector<rerun::Rgba32> colors;
  colors.reserve(cloud->size());
  for (const auto& p : cloud->points) {
    colors.emplace_back(p.r, p.g, p.b, 255);
  }
  auto point_cloud = rerun::Points3D(pts).with_colors(colors).with_radii({radius});
  std::string entity_path = entity + "/points";
  rec.log(entity_path, point_cloud);
  return entity_path;
}
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <pointcloud.pcd>" << std::endl;
    return -1;
  }
  std::string pcd_file = argv[1];
  if (!std::filesystem::exists(pcd_file)) {
    std::cerr << "File not found: " << pcd_file << std::endl;
    return -1;
  }
  std::cout << "Loading point cloud: " << pcd_file << std::endl;

  auto rec = rerun::RecordingStream("radial_extremum_detect_example");
  rec.spawn().exit_on_failure();

  ProfilerStart("profile.prof");

  using PointType = pcl::PointXYZRGB;

  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
  if (pcl::io::loadPLYFile(pcd_file, *cloud) == -1) {
    std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
    return -1;
  }
  std::cout << "Loaded point cloud with " << cloud->size() << " points." << std::endl;

  dklib::perception::pcproc::RadialSplitter<PointType> detector;
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();

  const Eigen::Vector3f offset = -origin;

  Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
  detector.setInputCloud(cloud);
  const float step = 10.0f;  // deg
  detector.setAngleStep(step);
  detector.setWidth(0.01f);
  detector.setCenter(offset);
  detector.setAxis(axis);

  auto start_time = std::chrono::high_resolution_clock::now();
  dklib::perception::detection::d3::RadialExtremumDetector<PointType> radial_detector(detector);
  auto [box, points_list] = radial_detector.execute();

  auto end_time = std::chrono::high_resolution_clock::now();
  ProfilerStop();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

  publishData(rec, "radial_extremum", box);
  publishData<PointType>(rec, "points", cloud);
  auto pl = radial_detector.getInlinePoints();
  publishData<PointType>(rec, "inline_points", pl);

  return 0;
}