#pragma once
#include <array>
#include <cstddef>

namespace dklib::perception::geometry {
/**
 * @brief Represents a triangle as a set of vertex indices.
 */
struct TriangleIndices {
  std::array<size_t, 3> vertex_indices;  ///< Indices of the triangle's vertices in a point cloud.

  std::array<std::pair<size_t, size_t>, 3> edge() const;

  size_t operator[](size_t index) const { return vertex_indices[index]; }
};

}  // namespace dklib::perception::geometry