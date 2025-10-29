#include <gperftools/profiler.h>
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

  voxblox::TsdfMap::Config config;
  config.tsdf_voxel_size = 0.02f;
  // config.tsdf_voxels_per_side = 1;
  voxblox::TsdfMap tsdf_map(config);

  voxblox::TsdfIntegratorBase::Config integrator_config;
  integrator_config.voxel_carving_enabled = true;
  integrator_config.default_truncation_distance = config.tsdf_voxel_size * 4.0f;
  auto integrator = voxblox::TsdfIntegratorFactory::create(voxblox::TsdfIntegratorType::kFast, integrator_config,
                                                           tsdf_map.getTsdfLayerPtr());

  voxblox::Transformation t_g_c;
  t_g_c.setIdentity();
  t_g_c.getPosition() = voxblox::Point(0.0, 0.0, 1.0);
  Eigen::Quaternionf q;
  q = Eigen::AngleAxisf(10.0 * M_PI / 180.0f, Eigen::Vector3f::UnitY());
  t_g_c.getRotation() = voxblox::Rotation{q.w(), q.x(), q.y(), q.z()};
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
#if 0
  if (pcl::io::loadPLYFile(pcd_file, *cloud) == -1) {
    std::cerr << "Failed to load point cloud: " << pcd_file << std::endl;
    return -1;
  }
#else
  // z = 1.0の平面上にx: -1 ~ 1.0, y: -1.0 ~ 1.0の点群を生成
  const float delta = 0.01f;
  for (float x = -1.0f; x <= 1.0f; x += delta) {
    for (float y = -1.0f; y <= 1.0f; y += delta) {
      PointType point;
      point.x = x;
      point.y = y;
      point.z = 0.5f;
      point.r = static_cast<uint8_t>((x + 1.0f) / 2.0f * 255);
      point.g = static_cast<uint8_t>((y + 1.0f) / 2.0f * 255);
      point.b = 0;
      cloud->points.push_back(point);
    }
  }
#endif

  publishData<PointType>(rec, "input_cloud", cloud, 0.005f);

  auto [points, colors] = dklib::perception::reconstruction::convertPointCloud<PointType>(*cloud);
  integrator->integratePointCloud(t_g_c, points, colors);
  pcl::PointCloud<PointType>::Ptr occupied_voxels(new pcl::PointCloud<PointType>());
  voxblox::createOccupancyBlocksFromTsdfLayer(tsdf_map.getTsdfLayer(), occupied_voxels);
  publishData<pcl::PointXYZRGB>(rec, "occupied_voxels", occupied_voxels, config.tsdf_voxel_size);

  pcl::PointCloud<pcl::PointXYZI>::Ptr icloud(new pcl::PointCloud<pcl::PointXYZI>());
  voxblox::createDistancePointcloudFromTsdfLayer(tsdf_map.getTsdfLayer(), icloud.get());
  publishVoxelData<pcl::PointXYZI>(rec, "tsdf", icloud, config.tsdf_voxel_size);

  return 0;
}