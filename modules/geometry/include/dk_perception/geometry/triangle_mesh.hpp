#pragma once
#include <Eigen/Core>
#include <dk_perception/geometry/triangle.hpp>
#include <vector>

namespace dklib::perception::geometry {
template <typename PointT>
class TriangleMesh {
 public:
  TriangleMesh() = default;
  TriangleMesh(const std::vector<PointT>& vertices) : vertices_(vertices) {}

  TriangleMesh(const std::vector<PointT>& vertices, const std::vector<TriangleIndices>& triangles)
      : vertices_(vertices), triangles_(triangles) {}

  void addTriangle(const std::array<size_t, 3>& triangle) {
    triangles_.emplace_back(TriangleIndices{triangle[0], triangle[1], triangle[2]});
  }

  void addTriangle(const TriangleIndices& triangle) { triangles_.push_back(triangle); }
  /**
   * @brief Returns the triangles in the mesh.
   * @return A vector of TriangleIndices.
   */
  const std::vector<TriangleIndices>& triangles() const { return triangles_; }

  const std::vector<PointT>& vertices() const { return vertices_; }

 private:
  std::vector<PointT> vertices_;            ///< The vertices of the mesh
  std::vector<TriangleIndices> triangles_;  ///< The triangles in the mesh.
};

}  // namespace dklib::perception::geometry
