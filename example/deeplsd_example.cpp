#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "dk_perception/detection/d2/triangle_constructor.hpp"
#include "dk_perception/dnn/deep_lsd.hpp"

int main(int argc, char* argv[]) {
  std::string model_path = std::string(argv[1]);
  std::string image_path = std::string(argv[2]);

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Could not read the image at " << image_path << std::endl;
    return -1;
  }

  // resize the image to 1280x720
  cv::resize(image, image, cv::Size(1280, 720), cv::INTER_AREA);
  if (image.channels() != 3) {
    std::cerr << "Error: Image must have 3 channels (BGR format)." << std::endl;
    return -1;
  }
  auto detector = std::make_shared<dklib::dnn::DeepFastLineSegmentDetector>(
      model_path, dklib::dnn::InferenceDevice::kCUDA, dklib::dnn::DeepLsdAttractionField::InputSize::kInputSize1280);

  std::vector<cv::Vec4f> lines;
  detector->detect(image, lines);
  dklib::perception::detection::d2::TriangleConstructorByLineIntersections triangle_constructor(detector);
  triangle_constructor.construct(image);

  for (const auto& line : lines) {
    cv::Point p1(line[0], line[1]);
    cv::Point p2(line[2], line[3]);
    cv::line(image, p1, p2, cv::Scalar(0, 255, 0), 1);  // Draw lines in green color
  }
  cv::imshow("Detected Lines", image);
  // cv::waitKey(0);

  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> line_segments;
  line_segments.reserve(lines.size());
  for (const auto& line : lines) {
    line_segments.emplace_back(Eigen::Vector2f(line[0], line[1]), Eigen::Vector2f(line[2], line[3]));
  }

  // auto right_angle_intersections = dklib::perception::detection::d2::findLineIntersections(line_segments, 5.0f,
  // 0.5f); std::cout << "Number of right-angle intersections found: " << right_angle_intersections.size() << std::endl;
  // for (const auto& intersection : right_angle_intersections) {
  //   const auto& key = intersection.first;
  //   const auto info = intersection.second;
  //   if (!info) {
  //     continue;
  //   }

  //   std::cout << "Intersection between line segments " << key.first << " and " << key.second
  //             << ": Intersection Point: (" << info->intersection_point.x() << ", " << info->intersection_point.y()
  //             << "), t1: " << info->t1 << ", t2: " << info->t2 << std::endl;
  //   cv::Point2f intersection_point(info->intersection_point.x(), info->intersection_point.y());
  //   cv::circle(image, intersection_point, 5, cv::Scalar(0, 0, 255), -1);  // Draw intersection points in red color
  // }
  // cv::imshow("Intersections", image);
  // cv::waitKey(0);
  return 0;
}