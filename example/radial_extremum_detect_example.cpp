#include <pcl/io/ply_io.h>

#include <dk_perception/detection/d3/radial_extremum_detector.hpp>
#include <filesystem>
#include <iostream>

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

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  if (pcl::io::loadPLYFile(pcd_file, *cloud) == -1) {
    std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
    return -1;
  }
  std::cout << "Loaded point cloud with " << cloud->size() << " points." << std::endl;

  std::filesystem::path output_dir = std::filesystem::path(pcd_file).parent_path() / "radial_extremum_output";
  std::filesystem::create_directories(output_dir);
  pcl::io::savePLYFile(output_dir / "input_cloud.ply", *cloud);

  dklib::perception::detection::d3::RadialExtremumDetector<pcl::PointXYZRGB> detector;
  detector.setInputCloud(cloud);
  detector.setAngleStep(45.0f);

  std::vector<float> angles;
  std::vector<pcl::PointIndices> point_indices;
  detector.detect(angles, point_indices);

  for (size_t i = 0; i < angles.size(); ++i) {
    std::cout << "Angle: " << angles[i] << " degrees, Points: " << point_indices[i].indices.size() << std::endl;
    if (point_indices[i].indices.empty()) {
      continue;
    }
    pcl::PointCloud<pcl::PointXYZRGB> subset;
    subset.reserve(point_indices[i].indices.size());
    for (int idx : point_indices[i].indices) {
      subset.push_back(cloud->at(idx));
    }
    pcl::io::savePLYFile(output_dir / ("angle_" + std::to_string(static_cast<int>(angles[i])) + ".ply"), subset);
  }
  return 0;
}