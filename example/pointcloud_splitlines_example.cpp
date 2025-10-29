#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <dk_perception/pcproc/scanline_splitter.hpp>
#include <dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

int main(int argc, char* argv[]) {
  cv::Mat rgb_img = cv::imread("/home/dean/workspace/src/dklib/.archives/scripts/rgbd_data/color.jpg");
  cv::Mat depth_img =
      cv::imread("/home/dean/workspace/src/dklib/.archives/scripts/rgbd_data/depth.png", cv::IMREAD_ANYDEPTH);
  std::string camera_info_file = "/home/dean/workspace/src/dklib/.archives/scripts/rgbd_data/rgb_camera_info.json";
  Eigen::Matrix3f intrinsic;
  intrinsic << 419.82196044921875, 0.0, 420.063720703125, 0.0, 418.7557067871094, 246.293212890625, 0.0, 0.0, 1.0;
  dklib::perception::type::pointcloud::DepthImageSet rgbd_image_set(rgb_img, depth_img, intrinsic, 0.001);

  dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<
      dklib::perception::type::pointcloud::DepthImageSet>
      accessor(rgbd_image_set);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  point_cloud->reserve(accessor.size());
  for (size_t i = 0; i < accessor.size(); i++) {
    pcl::PointXYZRGB point;
    Eigen::Vector3f pt = accessor.point_at(i);
    if (pt.array().isNaN().any()) {
      continue;
    }
    point.x = pt[0];
    point.y = pt[1];
    point.z = pt[2];
    auto color = accessor.color_at(i).value();
    point.b = color[0];
    point.g = color[1];
    point.r = color[2];
    point_cloud->emplace_back(point);
  }

  pcl::io::savePLYFile("pc.ply", *point_cloud);

  dklib::perception::pcproc::ScanlineSplitter<pcl::PointXYZRGB> splitter;
  using SplitAxis = dklib::perception::pcproc::ScanlineSplitter<pcl::PointXYZRGB>::SplitAxis;
  // split by scan lines
  auto regions = splitter.splitByScanLines(point_cloud, SplitAxis::kX);
  // apply color to each region, i % 2 == 0 ? red : green
  for (size_t i = 0; i < regions.size(); i++) {
    auto indices = regions[i];
    for (size_t j = 0; j < indices->indices.size(); j++) {
      if (i % 2 == 0) {
        point_cloud->at(indices->indices.at(j)).r = 255;
        point_cloud->at(indices->indices.at(j)).g = 0;
        point_cloud->at(indices->indices.at(j)).b = 0;
      } else {
        point_cloud->at(indices->indices.at(j)).r = 0;
        point_cloud->at(indices->indices.at(j)).g = 255;
        point_cloud->at(indices->indices.at(j)).b = 0;
      }
    }
  }

  pcl::io::savePLYFile("pc_split_x.ply", *point_cloud);

  // split by scan lines
  regions = splitter.splitByScanLines(point_cloud, SplitAxis::kY);
  // apply color to each region, i % 2 == 0 ? red : green
  for (size_t i = 0; i < regions.size(); i++) {
    auto indices = regions[i];
    for (size_t j = 0; j < indices->indices.size(); j++) {
      if (i % 2 == 0) {
        point_cloud->at(indices->indices.at(j)).r = 255;
        point_cloud->at(indices->indices.at(j)).g = 0;
        point_cloud->at(indices->indices.at(j)).b = 0;
      } else {
        point_cloud->at(indices->indices.at(j)).r = 0;
        point_cloud->at(indices->indices.at(j)).g = 255;
        point_cloud->at(indices->indices.at(j)).b = 0;
      }
    }
  }
  pcl::io::savePLYFile("pc_split_y.ply", *point_cloud);

  return 0;
}
