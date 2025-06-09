#define CATCH_CONFIG_MAIN

#include <Eigen/Core>
#include <catch2/catch.hpp>

#include "dk_perception/detection/d3/rectangle_detection.hpp"

TEST_CASE("exclude invalid triangles", "[exclude_invalid_triangles]") {
  using namespace dklib::perception::detection::d3;

  // Sample points and triangles
  std::vector<Eigen::Vector3f> points = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(0, 1, 0),
                                         Eigen::Vector3f(1, 1, 0)};

  std::vector<RightAngleTriangle> triangles = {{{0, 1, 2}, 0, Eigen::Vector3f(0, 0, 1)},
                                               {{1, 2, 3}, 2, Eigen::Vector3f(0, 0, 1)}};

  auto kdtree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->push_back({0.0, 0.5, 0.0});
  cloud->push_back({0.5, 0.0, 0.0});
  kdtree->setInputCloud(cloud);

  SECTION("Valid triangles remain after filtering") {
    excludeInvalidTriangles<pcl::PointXYZ>(points, triangles, kdtree, 2, 0.02, 0.1);
    REQUIRE(triangles.size() == 1);
    REQUIRE(triangles[0].vertex_indices[0] == 0);
    REQUIRE(triangles[0].vertex_indices[1] == 1);
    REQUIRE(triangles[0].vertex_indices[2] == 2);
    REQUIRE(triangles[0].right_angle_vertex_index == 0);
  }
}