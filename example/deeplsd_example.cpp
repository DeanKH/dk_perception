#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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
  dklib::dnn::DeepFastLineSegmentDetector detector(model_path, dklib::dnn::InferenceDevice::kCUDA,
                                                   dklib::dnn::DeepLsdAttractionField::InputSize::kInputSize1280);

  std::vector<cv::Vec4f> lines;
  detector.detect(image, lines);

  for (const auto& line : lines) {
    cv::Point p1(line[0], line[1]);
    cv::Point p2(line[2], line[3]);
    cv::line(image, p1, p2, cv::Scalar(0, 255, 0), 1);  // Draw lines in green color
  }
  cv::imshow("Detected Lines", image);
  cv::waitKey(0);
  return 0;
}