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

  dklib::dnn::DeepLsdAttractionField deep_lsd(model_path, dklib::dnn::InferenceDevice::kCUDA,
                                              dklib::dnn::DeepLsdAttractionField::InputSize::kInputSize1280);
  auto result = deep_lsd.inference(image);

  std::cout << "df_norm shape: " << result.df_norm.size() << std::endl;
  std::cout << "df shape: " << result.df.size() << std::endl;
  std::cout << "line_level shape: " << result.line_level.size() << std::endl;

  // normalize df_norm to [0, 1] range for visualization and convert to CV_8U
  cv::Mat df_norm_vis;
  cv::normalize(result.df_norm, df_norm_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::imshow("df_norm", df_norm_vis);

  cv::Mat df_vis;
  cv::normalize(result.df, df_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::imshow("df", df_vis);

  cv::Mat line_level_vis;
  cv::normalize(result.line_level, line_level_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::imshow("line_level", line_level_vis);

  cv::waitKey(0);
  return 0;
}