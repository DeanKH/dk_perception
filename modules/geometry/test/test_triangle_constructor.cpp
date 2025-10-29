#include <cmath>
#define CATCH_CONFIG_MAIN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <catch2/catch.hpp>

#include "dk_perception/geometry/triangle_constructor.hpp"

TEST_CASE("LineRef angle", "[LineRef]") {
  using namespace dklib::perception::geometry;

  // Create a vector of Line segments
  std::vector<Line<Eigen::Vector2f>> lines = {{Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(1.0f, 0.0f)},
                                              {Eigen::Vector2f(0.1f, -0.1f), Eigen::Vector2f(0.1f, 0.5f)}};

  auto [line_indices_list, points] = extractLineSegments(lines);
  auto l1 = line_indices_list[0].toLineRef(points);
  auto l2 = line_indices_list[1].toLineRef(points);
  float angle = l1.angle(l2);
  REQUIRE(angle == Approx(M_PI_2).margin(1e-6f));  // Expecting a right angle (90 degrees)
}

TEST_CASE("create triangle from Linesegments", "[calcTriangle]") {
  using namespace dklib::perception::geometry;

  // Create a vector of Line segments
  std::vector<Line<Eigen::Vector2f>> lines = {{Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(1.0f, 0.0f)},
                                              {Eigen::Vector2f(0.1f, -0.1f), Eigen::Vector2f(0.1f, 0.5f)}};

  // Extract line segments and points
  auto [line_indices_list, points] = extractLineSegments(lines);
  auto l1 = line_indices_list[0].toLineRef(points);
  auto l2 = line_indices_list[1].toLineRef(points);

  auto mesh = constructByLineIntersections<Eigen::Vector2f>(
      line_indices_list, points, [&points = points](const LineIndices& line1, const LineIndices& line2) {
        float angle = line1.angle(line2, points);

        if (angle > M_PI_2) {
          angle = M_PI - angle;  // Ensure angle is acute
        }

        return std::fabs(angle - M_PI_2) < 1e-3f;  // Check if angle is close to 90 degrees
      });

  REQUIRE(mesh.triangles().size() == 1);
  // REQUIRE(mesh.vertices().size() == 3);

  auto triangle = mesh.triangles()[0];
  Eigen::Vector2f v1 = mesh.vertices()[triangle[0]];
  Eigen::Vector2f v2 = mesh.vertices()[triangle[1]];
  Eigen::Vector2f v3 = mesh.vertices()[triangle[2]];
  std::cout << v1.transpose() << std::endl;
  std::cout << v2.transpose() << std::endl;
  std::cout << v3.transpose() << std::endl;
  // REQUIRE((v1 - lines[0].end).norm() < 1e-5f);
  // REQUIRE((v2 - lines[1].end).norm() < 1e-5f);
  // Eigen::Vector2f expected_v3{0.1f, 0.0f};
  // REQUIRE((v3 - expected_v3).norm() < 1e-5f);
}