#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/ply_io.h>
#include <voxblox/core/common.h>

#include <chrono>
#include <dk_perception/detection/d3/radial_extremum_detector.hpp>
#include <dk_perception/reconstruction/reconstruction.hpp>
#include <filesystem>
#include <iostream>
#include <rerun.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/recording_stream.hpp>

#include "rerun_publishers.hpp"

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

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

  publishData(rec, "box_opening", box);
  publishData<PointType>(rec, "points", cloud);
  auto pl = radial_detector.getInlinePoints();
  publishData<PointType>(rec, "inline_points", pl);

  std::cout << "box.center: " << box.center.transpose() << std::endl;
  std::cout << "box.size: " << box.size.transpose() << std::endl;
  std::cout << "box.orientation (w,x,y,z): " << box.orientation.w() << ", " << box.orientation.x() << ", "
            << box.orientation.y() << ", " << box.orientation.z() << std::endl;

  const Eigen::Matrix4d camera2boxtop_transform = box.getTransformation().cast<double>();
  const Eigen::Matrix4d boxtop2camera_transform = camera2boxtop_transform.inverse();

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  // pcl::transformPointCloud(*cloud, *transformed_cloud, boxtop2camera_transform.cast<float>());
  // publishData<PointType>(rec, "/box", transformed_cloud);

  double voxel_size = 0.02;
  dklib::perception::reconstruction::BoxInteriorReconstructor reconstructor(box, voxel_size);
  reconstructor.update<pcl::PointXYZRGB>(*cloud, Eigen::Matrix4f::Identity());
  const voxblox::Layer<voxblox::TsdfVoxel>& tsdf_layer = reconstructor.getTsdfLayer();

  pcl::PointCloud<pcl::PointXYZI>::Ptr icloud(new pcl::PointCloud<pcl::PointXYZI>());
  *icloud = reconstructor.getSdfVoxelInBox();
  publishData(rec, "box_opening/tsdf", Eigen::Isometry3d(camera2boxtop_transform));
  publishVoxelData<pcl::PointXYZI>(rec, "box_opening/tsdf", icloud, voxel_size);

  return 0;
}