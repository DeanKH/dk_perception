#pragma once
#include <cmath>
#include <dk_perception/geometry/intersection.hpp>
#include <dk_perception/geometry/line.hpp>
#include <dk_perception/geometry/triangle_mesh.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace dklib::perception::geometry {

template <typename PointT>
TriangleMesh<PointT> constructByLineIntersections(
    const std::vector<LineIndices>& line_indices_list, const std::vector<PointT>& points,
    std::function<bool(const LineIndices&, const LineIndices&)> filter_condition) {
  std::vector<PointT> vertices = points;
  // std::vector<PointT> intersection_points;
  std::vector<std::pair<float, float>> intersection_params;
  std::map<std::set<size_t>, size_t> intersection_map;
  std::vector<std::vector<size_t>> intersection_indices;
  intersection_indices.resize(line_indices_list.size());

  for (size_t i = 0; i < line_indices_list.size(); ++i) {
    for (size_t j = i + 1; j < line_indices_list.size(); ++j) {
      const auto& line1 = line_indices_list[i];
      const auto& line2 = line_indices_list[j];

      if (!filter_condition(line1, line2)) {
        std::cerr << "Skipping lines " << i << " and " << j << " due to filter condition." << std::endl;
        continue;
      }
      float t1 = std::numeric_limits<float>::quiet_NaN();
      float t2 = std::numeric_limits<float>::quiet_NaN();
      calcIntersection(points, line1, line2, t1, t2);
      if (std::isnan(t1) || std::isnan(t2)) {
        std::cerr << "No intersection found between lines " << i << " and " << j << "." << std::endl;
        continue;
      }

      // Store the intersection points
      auto intersect_point = line1.toLineRef(points).at(t1);
      vertices.push_back(intersect_point);
      intersection_params.emplace_back(t1, t2);
      intersection_map[{i, j}] = vertices.size() - 1;
      intersection_indices[i].push_back(j);
      intersection_indices[j].push_back(i);
    }
  }

  std::cout << "Found " << intersection_params.size() << " intersection points." << std::endl;

  const size_t point_num = points.size();
  TriangleMesh<PointT> mesh{vertices};
  for (const auto& [indices, idx] : intersection_map) {
    if (indices.size() != 2) {
      std::cerr << "Skipping invalid intersection indices: " << indices.size() << " indices found." << std::endl;
      continue;
    }
    size_t idx1 = *indices.begin();
    size_t idx2 = *std::next(indices.begin());
    const auto& intersection_param = intersection_params[idx - point_num];
    std::cout << intersection_param.first << " " << intersection_param.second << std::endl;
    std::array<size_t, 3> triangle_indices = {line_indices_list[idx1].far_idx(intersection_param.first),
                                              line_indices_list[idx2].far_idx(intersection_param.second), idx};
    mesh.addTriangle(triangle_indices);
  }
  return mesh;
}
}  // namespace dklib::perception::geometry