#include "dk_perception/detection/d2/node_extractor.hpp"

#include <Eigen/Core>
#include <fstream>

namespace {
cv::Mat readCSVtoMat(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<std::vector<float>> data;

  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stof(cell));
    }

    data.push_back(row);
  }

  // 行数と列数をチェック
  if (data.empty()) {
    std::cerr << "Empty or invalid CSV file.\n";
    return cv::Mat();
  }

  size_t rows = data.size();
  size_t cols = data[0].size();

  // OpenCVのMatに変換
  cv::Mat mat(rows, cols, CV_32F);
  for (size_t i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      std::cerr << "Inconsistent column size at row " << i << ".\n";
      return cv::Mat();
    }
    for (size_t j = 0; j < cols; ++j) {
      mat.at<float>(i, j) = data[i][j];
    }
  }

  return mat;
}

cv::Mat computeRowL2Norms(const cv::Mat& mat) {
  CV_Assert(mat.type() == CV_32F);  // float型限定

  int rows = mat.rows;
  cv::Mat norms(rows, 1, CV_32F);  // 結果：縦ベクトル（行数 × 1）

  for (int i = 0; i < rows; ++i) {
    cv::Mat row = mat.row(i);
    norms.at<float>(i, 0) = std::sqrt(row.dot(row));  // L2ノルム = √(row・row)
  }

  return norms;
}

Eigen::VectorXf computeRowL2Norms(const Eigen::MatrixXf& mat) { return mat.rowwise().norm(); }

// --- cv::Mat -> Eigen::MatrixXf 変換 ---
Eigen::MatrixXf cvMatToEigen(const cv::Mat& mat) {
  CV_Assert(mat.type() == CV_32F);  // float型前提
  Eigen::MatrixXf eigen_mat(mat.rows, mat.cols);
  for (int i = 0; i < mat.rows; ++i)
    for (int j = 0; j < mat.cols; ++j) eigen_mat(i, j) = mat.at<float>(i, j);
  return eigen_mat;
}

Eigen::MatrixXf computeSimilarityMatrix(const cv::Mat& cv_descriptors, const cv::Mat& cv_vocab) {
  // 1. 変換
  Eigen::MatrixXf descriptors = cvMatToEigen(cv_descriptors);  // NxD
  Eigen::MatrixXf vocab = cvMatToEigen(cv_vocab);              // MxD

  // 2. L2ノルム
  Eigen::VectorXf desc_norms = computeRowL2Norms(descriptors);  // Nx1
  Eigen::VectorXf vocab_norms = computeRowL2Norms(vocab);       // Mx1

  // 3. ノルムの外積 → MxN の正規化係数行列
  Eigen::MatrixXf norm_matrix = vocab_norms * desc_norms.transpose();  // MxN
  std::cout << "Norm matrix shape: " << norm_matrix.rows() << "x" << norm_matrix.cols() << std::endl;

  // 4. 内積計算 vocab * descriptors.T → MxN
  Eigen::MatrixXf dot_product = vocab * descriptors.transpose();  // MxN
  std::cout << "Dot product shape: " << dot_product.rows() << "x" << dot_product.cols() << std::endl;

  // 5. コサイン類似度 = 内積 / ノルム
  Eigen::MatrixXf similarity_matrix = dot_product.array() / norm_matrix.array();  // MxN

  // 6. 転置（元コードと合わせる）
  similarity_matrix.transposeInPlace();  // NxM
  std::cout << "Similarity matrix shape: " << similarity_matrix.rows() << "x" << similarity_matrix.cols() << std::endl;
  return similarity_matrix;
}
}  // namespace

namespace dklib::perception::detection::d2 {

NodeExtractorFromImageFeatures::NodeExtractorFromImageFeatures(
    const std::shared_ptr<dklib::experimental::SuperPoint>& superpoint, const std::string& vocab_dictionary_path)
    : superpoint_(superpoint) {
  vocab_ = readCSVtoMat(vocab_dictionary_path);
}

std::vector<cv::Point2f> NodeExtractorFromImageFeatures::extract(const cv::Mat& image) {
  if (!superpoint_) {
    throw std::runtime_error("SuperPoint model is not initialized.");
  }
  auto result = superpoint_->extract(image);
  Eigen::MatrixXf similarity_matrix = computeSimilarityMatrix(result.descriptors, vocab_);

  std::vector<cv::Point2f> keypoints;
  keypoints.reserve(result.keypoints.size());
  for (size_t i = 0; i < result.keypoints.size(); ++i) {
    if (roi.width > 0 && roi.height > 0) {
      if (!roi.contains(result.keypoints[i])) {
        continue;  // Skip keypoints outside the ROI
      }
    }

    if (result.scores[i] < min_score_) {
      continue;  // Skip keypoints with low scores
    }

    if (similarity_matrix.row(i).maxCoeff() >= similarity_threshold_) {
      keypoints.push_back(result.keypoints[i]);
    }
  }

  keypoints.shrink_to_fit();

  return keypoints;
}

}  // namespace dklib::perception::detection::d2