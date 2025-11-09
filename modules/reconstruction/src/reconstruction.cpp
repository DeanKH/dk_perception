#include <pcl/PolygonMesh.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <voxblox/core/color.h>
#include <voxblox/core/common.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/voxel.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/mesh/mesh_layer.h>

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

void setVertexColor(pcl::PointXYZRGB& pcl_point, const voxblox::Mesh::ConstPtr& mesh, size_t vertex_index,
                    int color_mode) {
  if (color_mode == 1) {
    pcl_point.r = mesh->colors[vertex_index].r;
    pcl_point.g = mesh->colors[vertex_index].g;
    pcl_point.b = mesh->colors[vertex_index].b;
  } else {
    constexpr float min_z = 0.0;
    constexpr float max_z = 3.0;
    const float z = mesh->vertices[vertex_index].z();
    const float d = std::clamp(z, min_z, max_z);
    const float h = (d - min_z) / (max_z - min_z);
    auto c = voxblox::rainbowColorMap(h);
    pcl_point.r = c.r;
    pcl_point.g = c.g;
    pcl_point.b = c.b;
  }
}

std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, std::vector<pcl::Vertices>> generateMesh(
    const voxblox::MeshLayer::Ptr& mesh_layer, int color_mode = 1) {
  assert(mesh_layer);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::vector<pcl::Vertices> polygons;

  voxblox::BlockIndexList mesh_indices;
  mesh_layer->getAllAllocatedMeshes(&mesh_indices);

  size_t vertex_index = 0;

  for (const voxblox::BlockIndex& block_index : mesh_indices) {
    voxblox::Mesh::ConstPtr mesh = mesh_layer->getMeshPtrByIndex(block_index);

    if (!mesh->hasVertices()) {
      continue;
    }

    // 頂点をpoint cloudに追加
    for (size_t i = 0; i < mesh->vertices.size(); i++) {
      pcl::PointXYZRGB point;
      point.x = mesh->vertices[i].x();
      point.y = mesh->vertices[i].y();
      point.z = mesh->vertices[i].z();

      // 色情報があれば設定
      if (mesh->hasColors()) {
        const voxblox::Color& color = mesh->colors[i];
        point.r = color.r;
        point.g = color.g;
        point.b = color.b;
      } else {
        point.r = point.g = point.b = 127;  // デフォルトグレー
      }

      cloud->push_back(point);
    }

    // 三角形を構成（マーチングキューブスの出力は三角形の頂点順）
    for (size_t i = 0; i < mesh->indices.size(); i += 3) {
      if (i + 2 < mesh->indices.size()) {
        pcl::Vertices triangle;
        triangle.vertices.resize(3);
        triangle.vertices[0] = vertex_index + mesh->indices[i];
        triangle.vertices[1] = vertex_index + mesh->indices[i + 1];
        triangle.vertices[2] = vertex_index + mesh->indices[i + 2];
        polygons.push_back(triangle);
      }
    }

    vertex_index += mesh->vertices.size();
  }

  // pcl::PolygonMesh に設定
  // pcl::toPCLPointCloud2(cloud, polygon_mesh->cloud);
  // polygon_mesh->polygons = polygons;

  return {cloud, polygons};
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

  integrator_ = voxblox::TsdfIntegratorFactory::create(voxblox::TsdfIntegratorType::kMerged, integrator_config,
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

  auto mesh_integrator_config = voxblox::MeshIntegratorConfig();
  mesh_layer_ = std::make_shared<voxblox::MeshLayer>(tsdf_map_->block_size());
  mesh_integrator_ = std::make_shared<voxblox::MeshIntegrator<voxblox::TsdfVoxel>>(
      mesh_integrator_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get());
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

  mesh_integrator_->generateMesh(false, false);
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

std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, std::vector<pcl::Vertices>> BoxInteriorReconstructor::generateMesh() {
  return ::generateMesh(mesh_layer_);
}

}  // namespace dklib::perception::reconstruction