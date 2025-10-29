#include <pcl/common/transforms.h>
#include <voxblox/core/common.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/voxel.h>
#include <voxblox/integrator/tsdf_integrator.h>

#include <dk_perception/reconstruction/reconstruction.hpp>

namespace {
template <typename FuncT>
double measureTime(FuncT&& func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  double mill_count = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Time taken: " << mill_count << " ms" << std::endl;
  return mill_count;
}

}  // namespace

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

  ::measureTime([&, &pointcloud = pointcloud, &colors = colors]() {
    std::cout << "integrating pointcloud" << std::endl;
    integrator_->integratePointCloud(T_L_C, pointcloud, colors);
    std::cout << "integration done" << std::endl;
  });

  ::measureTime([&]() {
    std::cout << "overwriting tsdf volume box boundary" << std::endl;
    overwriteTsdfVolumeBoxBoundary();
    std::cout << "overwrite done" << std::endl;
  });

  ::measureTime([&]() {
    std::cout << "cropping tsdf volume outside box" << std::endl;
    cropTsdfVolumeOutsideBox();
    std::cout << "cropping done" << std::endl;
  });

  ::measureTime([&]() {
    std::cout << "updating esdf from tsdf layer" << std::endl;
    esdf_integrator_->updateFromTsdfLayerBatch();
    std::cout << "esdf update done" << std::endl;
  });
}

/**
 * @brief boxの境界付近のTSDFボクセルを強制的に物体表面(distance = 0)にする.
 */
void BoxInteriorReconstructor::overwriteTsdfVolumeBoxBoundary() {
  voxblox::BlockIndexList blocks;
  auto layer = tsdf_map_->getTsdfLayerPtr();

  layer->getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = layer->voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  for (const auto& index : blocks) {
    // Iterate over all voxels in said blocks.
    auto& block = layer->getBlockByIndex(index);
    for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
      auto& voxel = block.getVoxelByLinearIndex(linear_index);
      const voxblox::Point& coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (coord.z() < 0.0) continue;
      if (!(voxel.weight > 0.0)) continue;
      // 中心が(0,0), sizeがbox_.sizeのboxの境界上にあるかどうかを確認
      if (std::abs(coord.x()) >= (box_.size.x() * 0.5f - voxel_size_ * 0.5f) &&
              std::abs(coord.x()) <= (box_.size.x() * 0.5f + voxel_size_ * 0.5f) ||
          std::abs(coord.y()) >= (box_.size.y() * 0.5f - voxel_size_ * 0.5f) &&
              std::abs(coord.y()) <= (box_.size.y() * 0.5f + voxel_size_ * 0.5f)) {
        voxel.distance = -voxel_size_ * 0.5f;
        voxel.weight = 100.0f;
      }
    }
  }
}

/**
 * @brief boxの外側にあるTSDFボクセルを削除する.
 */
void BoxInteriorReconstructor::cropTsdfVolumeOutsideBox() {
  voxblox::BlockIndexList blocks;
  auto layer = tsdf_map_->getTsdfLayerPtr();

  layer->getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = layer->voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;
  for (const auto& index : blocks) {
    // Iterate over all voxels in said blocks.
    auto& block = layer->getBlockByIndex(index);
    for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
      auto& voxel = block.getVoxelByLinearIndex(linear_index);
      const voxblox::Point& coord = block.computeCoordinatesFromLinearIndex(linear_index);
      // 中心が(0,0), sizeがbox_.sizeのboxの外側にあるかどうかを確認
      if (coord.z() < 0.0 || std::abs(coord.x()) > (box_.size.x() * 0.5f + voxel_size_ * 0.5f) ||
          std::abs(coord.y()) > (box_.size.y() * 0.5f + voxel_size_ * 0.5f)) {
        voxel.weight = 0.0f;  // ボクセルを無効化
      }
    }
  }
}
}  // namespace dklib::perception::reconstruction