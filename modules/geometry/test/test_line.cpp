#define CATCH_CONFIG_MAIN
#include <Eigen/Core>
#include <catch2/catch.hpp>

#include "dk_perception/geometry/line.hpp"

TEST_CASE("LineRef vector calculation", "[LineRef]") {
  using namespace dklib::perception::geometry;

  // Create a line segment
  LineRef<Eigen::Vector2f> line_ref{Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(1.0f, 1.0f)};

  // Calculate the vector of the line segment
  Eigen::Vector2f vector = line_ref.vector();

  // Check if the vector is correct
  REQUIRE(vector.x() == Approx(1.0f).margin(1e-6));
  REQUIRE(vector.y() == Approx(1.0f).margin(1e-6));
}

TEST_CASE("create LineIndices from Linesegments", "[extractLineSegments]") {
  using namespace dklib::perception::geometry;

  // Create a vector of Line segments
  std::vector<Line<Eigen::Vector2f>> lines = {{Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(1.0f, 1.0f)},
                                              {Eigen::Vector2f(1.0f, 1.0f), Eigen::Vector2f(2.0f, 2.0f)}};

  // Extract line segments and points
  auto [line_indices_list, points] = extractLineSegments(lines);

  // Check the size of the line indices list
  REQUIRE(line_indices_list.size() == lines.size());
  REQUIRE(points.size() == lines.size() * 2);

  // Check the first line indices
  REQUIRE(line_indices_list[0].start_idx == 0);
  REQUIRE(line_indices_list[0].end_idx == 1);
}