#include <pcl/common/transforms.h>
#include <voxblox/core/common.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>

#include <dk_perception/reconstruction/reconstruction.hpp>

namespace dklib::perception::reconstruction {
BoxInteriorReconstructor::BoxInteriorReconstructor(const geometry::BoundingBox3D& box, const float voxel_size)
    : box_{box}, voxel_size_{voxel_size} {
  voxblox::TsdfMap::Config config;
  config.tsdf_voxel_size = voxel_size_;
  tsdf_map_ = std::make_shared<voxblox::TsdfMap>(config);

  std::cout << "Initialized TSDF Map with voxel size: " << config.tsdf_voxel_size
            << " and voxels per side: " << config.tsdf_voxels_per_side << std::endl;

  voxblox::TsdfIntegratorBase::Config integrator_config;
  integrator_config.voxel_carving_enabled = true;
  integrator_config.default_truncation_distance = config.tsdf_voxel_size * 4.0f;

  integrator_ = voxblox::TsdfIntegratorFactory::create(voxblox::TsdfIntegratorType::kFast, integrator_config,
                                                       tsdf_map_->getTsdfLayerPtr());

  voxblox::EsdfMap::Config esdf_config;
  esdf_config.esdf_voxel_size = config.tsdf_voxel_size;
  esdf_config.esdf_voxels_per_side = config.tsdf_voxels_per_side;

  esdf_map_ = std::make_shared<voxblox::EsdfMap>(esdf_config);
  auto esdf_integrator_config = voxblox::EsdfIntegrator::Config();
  esdf_integrator_config.min_distance_m = integrator_config.default_truncation_distance / 2.0;
  esdf_integrator_config.full_euclidean_distance = false;

  esdf_integrator_.reset(
      new voxblox::EsdfIntegrator(esdf_integrator_config, tsdf_map_->getTsdfLayerPtr(), esdf_map_->getEsdfLayerPtr()));
}

template <>
void BoxInteriorReconstructor::update(const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                      const Eigen::Matrix4f& cloud2boxbase_transform) {
  const Eigen::Matrix4d camera2boxtop_transform =
      cloud2boxbase_transform.cast<double>() * box_.getTransformation().cast<double>();
  const Eigen::Matrix4d boxtop2camera_transform = camera2boxtop_transform.inverse();

  auto [pointcloud, colors] = convertPointCloud<pcl::PointXYZRGB>(cloud);

  Eigen::Quaterniond boxtop2camera_orientation(
      boxtop2camera_transform.block<3, 3>(0, 0));  // Rotation part to quaternion
  Eigen::Vector3d boxtop2camera_position = boxtop2camera_transform.block<3, 1>(0, 3);

  voxblox::Transformation T_L_C;
  T_L_C.setIdentity();
  const voxblox::Rotation r{boxtop2camera_orientation.cast<float>().w(), boxtop2camera_orientation.cast<float>().x(),
                            boxtop2camera_orientation.cast<float>().y(), boxtop2camera_orientation.cast<float>().z()};
  T_L_C.getPosition() = boxtop2camera_position.cast<float>();
  T_L_C.getRotation() = r;
  std::cout << "boxtop2camera_pos: " << boxtop2camera_position.cast<float>().transpose() << std::endl;

  integrator_->integratePointCloud(T_L_C, pointcloud, colors);

  esdf_integrator_->updateFromTsdfLayerBatch();
}

}  // namespace dklib::perception::reconstruction