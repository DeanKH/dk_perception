#pragma once
#include <Eigen/Geometry>
#include <memory>
#include <opencv2/aruco.hpp>
#include <optional>

namespace dklib::perception::detection::d3 {
enum class MarkerTargetType { ARUCO = 0, APRILTAG, UNKNOWN };

/*
 * ## constructor
 * - marker target type
 * ## input
 * - image: cv::Mat
 * - marker id
 * - marker length[m]
 * ## output
 * - 3d pose: std::optional<Eigen::Isometry3d>
 */
class MarkerTarget3DPoseDetector {
 public:
  virtual ~MarkerTarget3DPoseDetector() = default;
  virtual std::optional<Eigen::Isometry3d> detectMarkerWithID(
      const cv::Mat& image, const int marker_id, const float marker_length_m, const Eigen::Matrix3d& camera_intrinsics,
      const std::array<double, 5>& camera_distortion_coeffs) = 0;
};

class ArucoMarkerTarget3DPoseDetector : public MarkerTarget3DPoseDetector {
 public:
  struct Param {
    int dict_id = cv::aruco::DICT_4X4_50;
  };

  ArucoMarkerTarget3DPoseDetector(const Param& param);

  std::optional<Eigen::Isometry3d> detectMarkerWithID(const cv::Mat& image, const int marker_id,
                                                      const float marker_length_m,
                                                      const Eigen::Matrix3d& camera_intrinsics,
                                                      const std::array<double, 5>& camera_distortion_coeffs) override;

 private:
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
};

template <typename MarkerArgType>
std::shared_ptr<MarkerTarget3DPoseDetector> createMarkerDetector(const MarkerTargetType& marker_type,
                                                                 const MarkerArgType& arg) {
  switch (marker_type) {
    case MarkerTargetType::ARUCO:
      return std::make_shared<ArucoMarkerTarget3DPoseDetector>(arg);
    // case MarkerTargetType::APRILTAG:
    //   return std::make_shared<AprilTagMarkerTarget3DPoseDetector>();
    default:
      throw std::runtime_error("Unsupported marker target type");
  }
}

}  // namespace dklib::perception::detection::d3