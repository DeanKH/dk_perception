#include "dk_perception/detection/d2/triangle_constructor.hpp"

#include <Eigen/Core>

namespace dklib::perception::detection::d2 {
dklib::perception::type::Polygons<Eigen::Vector2f> TriangleConstructorByLineIntersections::construct(
    const cv::Mat& image) {
  // Detect line segments in the image
  std::vector<cv::Vec4f> lines;  // x1, y1, x2, y2
  line_segment_detector_->detect(image, lines);
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> line_segments;
  line_segments.reserve(lines.size());

  for (const auto& line : lines) {
    line_segments.emplace_back(Eigen::Vector2f(line[0], line[1]), Eigen::Vector2f(line[2], line[3]));
  }

  auto right_angle_intersections = findLineIntersections(line_segments, 10.0f, 10.0f);
  std::map<size_t, std::list<std::shared_ptr<IntersectionInfo>>> intersection_map;
  for (const auto& intersection : right_angle_intersections) {
    const auto& key = intersection.first;
    const auto info = intersection.second;
    if (!info) {
      continue;
    }
    intersection_map[key.first].push_back(info);
    intersection_map[key.second].push_back(info);
  }
  // TODO: ここまでクラスの外に移動する

  // Process the detected line segments to form triangles
  dklib::perception::type::Polygons<Eigen::Vector2f> triangles;
  // triangles.points.reserve(right_angle_intersections.size());
  // for (const auto& [key, value] : right_angle_intersections) {
  //   const auto& info = value;
  //   if (!info) {
  //     continue;
  //   }
  //   triangles.points.emplace_back(info->intersection_point.x(), info->intersection_point.y());
  // }

  {
    size_t count = 0;
    for (const auto& [key, value] : right_angle_intersections) {
      auto l1 = line_segments[key.first];
      auto l2 = line_segments[key.second];

      // triangles.polygons.emplace_back();
    }
  }

  return triangles;
}

}  // namespace dklib::perception::detection::d2
