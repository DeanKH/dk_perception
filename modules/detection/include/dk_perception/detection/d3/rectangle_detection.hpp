#pragma once
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>

#include <concepts>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>
#include <dk_perception/type/pointcloud/rgbd_type.hpp>
#include <opencv2/core.hpp>

namespace dklib::perception::detection::d3 {

template <class T>
concept MeshReconstructable = requires(T& x, pcl::PolygonMesh& mesh) {
  x.reconstruct(mesh);
};

template <MeshReconstructable T>
class RectangleDetection {
 public:
  RectangleDetection() {}
  ~RectangleDetection() {}

  void detect(const dklib::perception::type::pointcloud::DepthImageSet& rgbd) {
    dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<
        dklib::perception::type::pointcloud::DepthImageSet>
        accessor(rgbd);
  }
};
}  // namespace dklib::perception::detection::d3