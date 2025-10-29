#include <dk_perception/detection/d3/marker_target_detection.hpp>
#include <opencv2/calib3d.hpp>

namespace dklib::perception::detection::d3 {
ArucoMarkerTarget3DPoseDetector::ArucoMarkerTarget3DPoseDetector(const Param& param) {
  dictionary_ = cv::aruco::getPredefinedDictionary(param.dict_id);
}

std::optional<Eigen::Isometry3d> ArucoMarkerTarget3DPoseDetector::detectMarkerWithID(
    const cv::Mat& image, const int marker_id, const float marker_length_m, const Eigen::Matrix3d& camera_intrinsics,
    const std::array<double, 5>& camera_distortion_coeffs) {
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;

  cv::aruco::detectMarkers(image, dictionary_, marker_corners, marker_ids);
  // cv::aruco::refineDetectedMarkers(image, dictionary_, marker_corners, marker_ids, rejected_candidates);

  if (marker_ids.empty()) {
    return std::nullopt;
  }

  // Get the first detected marker's corners
  const auto& corners = marker_corners[0];

  // Estimate the pose of the marker
  Eigen::Isometry3d pose;
  pose.setIdentity();
  std::vector<cv::Vec3d> rvecs, tvecs;
  cv::Mat camera_matrix(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      camera_matrix.at<double>(i, j) = camera_intrinsics(i, j);
    }
  }

  cv::aruco::estimatePoseSingleMarkers(marker_corners, marker_length_m, camera_matrix,
                                       cv::Mat(camera_distortion_coeffs), rvecs, tvecs);
  if (rvecs.empty() || tvecs.empty()) {
    return std::nullopt;
  }

  Eigen::Quaterniond q = [&]() {
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvecs[0], rotation_matrix);

    Eigen::Matrix3d R_eigen;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R_eigen(i, j) = rotation_matrix.at<double>(i, j);
      }
    }
    return Eigen::Quaterniond(R_eigen);
  }();

  Eigen::Vector3d t;
  t << tvecs[0][0], tvecs[0][1], tvecs[0][2];

  pose = Eigen::Isometry3d::Identity();
  pose.rotate(q);
  pose.pretranslate(t);

  return pose;
}

}  // namespace dklib::perception::detection::d3