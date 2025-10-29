#include "dk_perception/optimization/rectangle_2d_fitting.hpp"

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <cmath>

namespace
{

struct BoundingBox2D
{
  Eigen::Vector2f center;
  Eigen::Vector2f size;
  float rotation;  // in radians
};

/// 入力点ptに対して、矩形bboxの境界上で最も近い点を計算する
Eigen::Vector2d closestPointOnBoundingBox2dBoundary(
  const Eigen::Vector2d & pt,
  const BoundingBox2D & bbox)
{
  // Convert to double precision for calculations
  Eigen::Vector2d center = bbox.center.cast<double>();
  Eigen::Vector2d size = bbox.size.cast<double>();
  double theta = bbox.rotation;

  // Create rotation matrix (inverse rotation to transform to local coords)
  Eigen::Matrix2d R_inv;
  R_inv << cos(-theta), -sin(-theta), sin(-theta), cos(-theta);

  // Transform point to local coordinate system (centered at origin, no
  // rotation)
  Eigen::Vector2d pt_local = R_inv * (pt - center);

  // Half sizes
  double half_w = size.x() * 0.5;
  double half_h = size.y() * 0.5;

  // Clamp to rectangle boundary in local coordinates
  Eigen::Vector2d closest_local;
  closest_local.x() = std::max(-half_w, std::min(half_w, pt_local.x()));
  closest_local.y() = std::max(-half_h, std::min(half_h, pt_local.y()));

  // If point is inside rectangle, project to closest boundary
  if (std::abs(pt_local.x()) <= half_w && std::abs(pt_local.y()) <= half_h) {
    // Calculate distances to each boundary
    double dist_left = pt_local.x() + half_w;
    double dist_right = half_w - pt_local.x();
    double dist_bottom = pt_local.y() + half_h;
    double dist_top = half_h - pt_local.y();

    // Find minimum distance and project to that boundary
    double min_dist = std::min({dist_left, dist_right, dist_bottom, dist_top});

    if (min_dist == dist_left) {
      closest_local.x() = -half_w;
    } else if (min_dist == dist_right) {
      closest_local.x() = half_w;
    } else if (min_dist == dist_bottom) {
      closest_local.y() = -half_h;
    } else {
      closest_local.y() = half_h;
    }
  }

  // Transform back to global coordinates
  Eigen::Matrix2d R;
  R << cos(theta), -sin(theta), sin(theta), cos(theta);

  return center + R * closest_local;
}

}  // namespace

namespace dklib::perception::optimization
{
Rectangle2dPointFactor::Rectangle2dPointFactor(
  gtsam::Key pose_key, gtsam::Key size_key, const gtsam::Point2 & point,
  const gtsam::SharedNoiseModel & noise_model)
: gtsam::NoiseModelFactorN<gtsam::Pose2, gtsam::Point2>(noise_model,
    pose_key, size_key),
  point_(point) {}

gtsam::Vector Rectangle2dPointFactor::evaluateError(
  const gtsam::Pose2 & pose, const gtsam::Point2 & size,
  gtsam::OptionalMatrixType H1, gtsam::OptionalMatrixType H2) const
{
  Eigen::Vector2d center(pose.x(), pose.y());
  double theta = pose.theta();

  BoundingBox2D bbox;
  bbox.center = Eigen::Vector2f(center.x(), center.y());
  bbox.size = Eigen::Vector2f(size.x(), size.y());
  bbox.rotation = theta;

  Eigen::Vector2d point(point_.x(), point_.y());
  Eigen::Vector2d q_on_rect = closestPointOnBoundingBox2dBoundary(point, bbox);

  Eigen::Vector2d err = point - q_on_rect;

  // Compute Jacobians using finite differences
  if (H1) {
    const double delta = 1e-5;
    gtsam::Matrix J1(2, 3);

    // df/dx
    BoundingBox2D bbox_dx = bbox;
    bbox_dx.center.x() += delta;
    Eigen::Vector2d q_dx = closestPointOnBoundingBox2dBoundary(point, bbox_dx);
    Eigen::Vector2d err_dx = point - q_dx;
    J1.col(0) = (err_dx - err) / delta;

    // df/dy
    BoundingBox2D bbox_dy = bbox;
    bbox_dy.center.y() += delta;
    Eigen::Vector2d q_dy = closestPointOnBoundingBox2dBoundary(point, bbox_dy);
    Eigen::Vector2d err_dy = point - q_dy;
    J1.col(1) = (err_dy - err) / delta;

    // df/dtheta
    BoundingBox2D bbox_dtheta = bbox;
    bbox_dtheta.rotation += delta;
    Eigen::Vector2d q_dtheta =
      closestPointOnBoundingBox2dBoundary(point, bbox_dtheta);
    Eigen::Vector2d err_dtheta = point - q_dtheta;
    J1.col(2) = (err_dtheta - err) / delta;

    *H1 = J1;
  }

  if (H2) {
    const double delta = 1e-5;
    gtsam::Matrix J2(2, 2);

    // df/dwidth
    BoundingBox2D bbox_dw = bbox;
    bbox_dw.size.x() += delta;
    Eigen::Vector2d q_dw = closestPointOnBoundingBox2dBoundary(point, bbox_dw);
    Eigen::Vector2d err_dw = point - q_dw;
    J2.col(0) = (err_dw - err) / delta;

    // df/dheight
    BoundingBox2D bbox_dh = bbox;
    bbox_dh.size.y() += delta;
    Eigen::Vector2d q_dh = closestPointOnBoundingBox2dBoundary(point, bbox_dh);
    Eigen::Vector2d err_dh = point - q_dh;
    J2.col(1) = (err_dh - err) / delta;

    *H2 = J2;
  }

  gtsam::Vector e(2);
  e << err.x(), err.y();
  return e;
}

Rectangle2dFitting::Rectangle2dFitting() {}

std::pair<gtsam::Pose2, gtsam::Point2> Rectangle2dFitting::Optimize(
  const std::vector<Eigen::Vector2f> & points,
  const gtsam::Pose2 & initial_pose,
  const gtsam::Point2 & initial_size)
{
  // 因子グラフの構築
  gtsam::NonlinearFactorGraph graph;

  // 各点に対してファクターを追加
  auto noise = gtsam::noiseModel::Isotropic::Sigma(2, 0.03);
  for (const auto & point : points) {
    gtsam::Point2 gtsam_point(point.x(), point.y());
    graph.emplace_shared<Rectangle2dPointFactor>(
      kPoseKey, kSizeKey, gtsam::Point2(point.x(), point.y()), noise);
  }

  auto prior_pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << 10.0, 10.0, 1.0).finished());
  graph.add(
    gtsam::PriorFactor<gtsam::Pose2>(
      kPoseKey, initial_pose,
      prior_pose_noise));

  // サイズが負にならないように事前分布を追加
  auto size_prior_noise = gtsam::noiseModel::Isotropic::Sigma(2, 10.0);
  graph.add(
    gtsam::PriorFactor<gtsam::Point2>(
      kSizeKey, initial_size,
      size_prior_noise));

  // 初期値の設定
  gtsam::Values initial_values;
  initial_values.insert(kPoseKey, initial_pose);
  initial_values.insert(kSizeKey, initial_size);

  // 最適化の実行
  gtsam::LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");
  params.setVerbosity("TERMINATION");
  params.setRelativeErrorTol(1e-08);
  params.setAbsoluteErrorTol(1e-08);
  params.maxIterations = 100;

  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values, params);
  gtsam::Values result = optimizer.optimize();

  // 結果の取得
  gtsam::Pose2 optimized_pose = result.at<gtsam::Pose2>(kPoseKey);
  gtsam::Point2 optimized_size = result.at<gtsam::Point2>(kSizeKey);

  return {optimized_pose, optimized_size};
}

}  // namespace dklib::perception::optimization
