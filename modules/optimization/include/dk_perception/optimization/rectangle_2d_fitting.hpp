#pragma once

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace dklib::perception::optimization
{

// 入力した2次元の座標を持つ点列を2D矩形にフィッティングする
class Rectangle2dPointFactor
  : public gtsam::NoiseModelFactorN<gtsam::Pose2, gtsam::Point2>
{
public:
  Rectangle2dPointFactor(
    gtsam::Key pose_key, gtsam::Key size_key,
    const gtsam::Point2 & point,
    const gtsam::SharedNoiseModel & noise_model);

  gtsam::Vector evaluateError(
    const gtsam::Pose2 & pose, const gtsam::Point2 & size,
    gtsam::OptionalMatrixType H1 = nullptr,
    gtsam::OptionalMatrixType H2 = nullptr) const override;

private:
  gtsam::Point2 point_;
};

// 矩形フィッティングの最適化問題を構築するヘルパークラス
class Rectangle2dFitting
{
public:
  Rectangle2dFitting();

  // 点列から矩形を最適化
  std::pair<gtsam::Pose2, gtsam::Point2> Optimize(
    const std::vector<Eigen::Vector2f> & points,
    const gtsam::Pose2 & initial_pose, const gtsam::Point2 & initial_size);

private:
  gtsam::Key kPoseKey = gtsam::Symbol('x', 0);
  gtsam::Key kSizeKey = gtsam::Symbol('s', 0);
};

}  // namespace dklib::perception::optimization
