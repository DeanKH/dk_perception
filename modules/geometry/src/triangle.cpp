#include <dk_perception/geometry/triangle.hpp>

namespace dklib::perception::geometry {

std::array<std::pair<size_t, size_t>, 3> TriangleIndices::edge() const {
  // Returns the edges of the triangle as pairs of vertex indices.
  return {std::make_pair(vertex_indices[0], vertex_indices[1]), std::make_pair(vertex_indices[1], vertex_indices[2]),
          std::make_pair(vertex_indices[2], vertex_indices[0])};
}

}  // namespace dklib::perception::geometry