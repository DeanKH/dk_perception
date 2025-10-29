#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>

namespace voxblox {
inline bool visualizeDistanceIntensityTsdfVoxels(const TsdfVoxel& voxel, const Point& /*coord*/, double* intensity) {
  assert(intensity);
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight) {
    *intensity = voxel.distance;
    return true;
  }
  return false;
}

/**
 * For intensities values, such as distances, which are mapped to a color only
 * by the subscriber.
 */
template <typename VoxelType>
using ShouldVisualizeVoxelIntensityFunctionType =
    std::function<bool(const VoxelType& voxel, const Point& coord, double* intensity)>;

/// Template function to visualize an intensity pointcloud.
template <typename VoxelType>
void createColorPointcloudFromLayer(const Layer<VoxelType>& layer,
                                    const ShouldVisualizeVoxelIntensityFunctionType<VoxelType>& vis_function,
                                    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  assert(pointcloud);
  pointcloud->clear();
  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  double intensity = 0.0;
  // Iterate over all blocks.
  for (const BlockIndex& index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block; ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord, &intensity)) {
        pcl::PointXYZI point;
        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();
        point.intensity = intensity;
        pointcloud->push_back(point);
      }
    }
  }
}

inline void createDistancePointcloudFromTsdfLayer(const Layer<TsdfVoxel>& layer,
                                                  pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  assert(pointcloud);
  createColorPointcloudFromLayer<TsdfVoxel>(layer, &visualizeDistanceIntensityTsdfVoxels, pointcloud);
}
}  // namespace voxblox
