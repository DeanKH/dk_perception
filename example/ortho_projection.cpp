// Copyright (c) 2025 deankh. All rights reserved.
#include <cmath>
#include <functional>
#include <iostream>
#include "dk_perception/type/pointcloud/iteratable_colorized_pointcloud_accessor.hpp"
#include "dk_perception/type/pointcloud/rgbd_type.hpp"
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "dk_perception/image_processing/percepective_to_ortho.hpp"
#include <pcl/impl/point_types.hpp>
#include <pcl/io/ply_io.h>
#include <algorithm>
#include <numeric>

std::vector<double> calculateAverageDistanceFromPoint(
  const cv::Mat & labels,
  const cv::Point & point, int nLabels)
{
  std::vector<double> average_distances(nLabels, 0.0);
  std::vector<int> label_counts(nLabels, 0);

  for (int y = 0; y < labels.rows; ++y) {
    for (int x = 0; x < labels.cols; ++x) {
      int label = labels.at<int>(y, x);
      if (label > 0) { // Ignore background
        double distance = std::sqrt(std::pow(x - point.x, 2) + std::pow(y - point.y, 2));
        average_distances[label] += distance;
        label_counts[label]++;
      }
    }
  }

  for (int i = 1; i < nLabels; ++i) {
    if (label_counts[i] > 0) {
      average_distances[i] /= label_counts[i];
    }
  }

  return average_distances;
}
// 各線分の方向ベクトルと長さを計算
struct LineInfo
{
  Eigen::Vector2f direction;       // 正規化された方向ベクトル
  float length;                    // 線分の長さ
  float angle;                     // X軸に対する角度 (度)
  int index;                       // 元の線分のインデックス
};

void drawRotatedRect(
  cv::Mat & image, const cv::RotatedRect & rotated_rect, const cv::Scalar & color,
  int thickness = 2)
{
  cv::Point2f vertices[4];
  rotated_rect.points(vertices);

  for (int i = 0; i < 4; ++i) {
    cv::line(image, vertices[i], vertices[(i + 1) % 4], color, thickness);
  }

  // 回転矩形の中心を表示
  cv::circle(image, rotated_rect.center, 3, color, -1);

  // 回転矩形の情報を表示
  std::stringstream ss;
  ss << "Size: " << rotated_rect.size.width << "x" << rotated_rect.size.height;
  ss << " Angle: " << rotated_rect.angle;
  cv::putText(
    image, ss.str(), cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

bool findLineIntersection(
  const std::pair<Eigen::Vector2f, Eigen::Vector2f> & line1,
  const std::pair<Eigen::Vector2f, Eigen::Vector2f> & line2,
  Eigen::Vector2f & intersection)
{
  Eigen::Vector2f p1 = line1.first;
  Eigen::Vector2f p2 = line1.second;
  Eigen::Vector2f p3 = line2.first;
  Eigen::Vector2f p4 = line2.second;

  float x1 = p1.x(), y1 = p1.y();
  float x2 = p2.x(), y2 = p2.y();
  float x3 = p3.x(), y3 = p3.y();
  float x4 = p4.x(), y4 = p4.y();

  float denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);

  // 線が平行の場合
  if (std::abs(denominator) < 1e-5f) {
    return false;
  }

  float ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator;
  float ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator;

  // 交差しているが線分上ではない場合
  if (ua < -2.0f || ua > 2.0f || ub < -2.0f || ub > 2.0f) {
    // 少し余裕を持たせる（0.1）
    return false;
  }

  // 交点の計算
  intersection.x() = x1 + ua * (x2 - x1);
  intersection.y() = y1 + ua * (y2 - y1);

  return true;
}

/**
 * @brief 4本の線分から最も妥当な回転矩形を計算する関数
 * @param lines 4本の線分（始点と終点のペア）
 * @return 計算された回転矩形
 */
cv::RotatedRect calculateRotatedRectFromLines(
  const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> & lines)
{
  if (lines.size() != 4) {
    std::cerr << "Expected 4 lines, got " << lines.size() << std::endl;
    return cv::RotatedRect();
  }

  // 各線分に対して、他の線分との交点を計算
  std::vector<cv::Point2f> corners;

  // 各線分のペアを調べる
  for (size_t i = 0; i < lines.size(); ++i) {
    for (size_t j = i + 1; j < lines.size(); ++j) {
      Eigen::Vector2f intersection;
      if (findLineIntersection(lines[i], lines[j], intersection)) {
        corners.push_back(cv::Point2f(intersection.x(), intersection.y()));
      }
    }
  }

  // 十分な交点が見つからない場合
  if (corners.size() < 3) {
    std::cerr << "Not enough intersection points: " << corners.size() << std::endl;
    return cv::RotatedRect();
  }

  // 重複する交点を除去
  for (size_t i = 0; i < corners.size(); ++i) {
    for (size_t j = i + 1; j < corners.size(); ) {
      float dist = std::sqrt(
        std::pow(corners[i].x - corners[j].x, 2) +
        std::pow(corners[i].y - corners[j].y, 2));
      if (dist < 5.0f) {  // 5ピクセル以内なら重複とみなす
        corners.erase(corners.begin() + j);
      } else {
        ++j;
      }
    }
  }

  // 最低4つの交点が必要
  if (corners.size() < 4) {
    // 見つかった交点が少ない場合、仮想的な点を追加して矩形を形成
    // 例えば、対角線上に点を補完する
    if (corners.size() == 3) {
      cv::Point2f center = (corners[0] + corners[1] + corners[2]) * (1.0f / 3.0f);
      // 中心点から最も遠い点の反対側に仮想点を配置
      float max_dist = 0;
      int farthest_idx = 0;
      for (int i = 0; i < 3; ++i) {
        float dist = std::sqrt(
          std::pow(corners[i].x - center.x, 2) +
          std::pow(corners[i].y - center.y, 2));
        if (dist > max_dist) {
          max_dist = dist;
          farthest_idx = i;
        }
      }
      cv::Point2f virtual_point = 2 * center - corners[farthest_idx];
      corners.push_back(virtual_point);
    }
  }

  // 凸包を計算して矩形の候補点を整理
  std::vector<cv::Point2f> hull;
  cv::convexHull(corners, hull);

  // 回転矩形を計算
  cv::RotatedRect rotatedRect = cv::minAreaRect(hull);

  return rotatedRect;
}

std::map<int, std::vector<LineInfo>> clusterLinesByAngle(
  const std::vector<LineInfo> & line_infos,
  float angle_tolerance_deg = 10.0f)
{
  if (line_infos.empty()) {
    return {};
  }

  // 角度でソートした線分情報のコピーを作成
  std::vector<LineInfo> sorted_lines = line_infos;
  std::sort(
    sorted_lines.begin(), sorted_lines.end(),
    [](const LineInfo & a, const LineInfo & b) {return a.angle < b.angle;});

  // クラスタリング結果
  std::map<int, std::vector<LineInfo>> clusters;

  int current_cluster = 0;
  float current_angle = sorted_lines[0].angle;
  clusters[current_cluster].push_back(sorted_lines[0]);

  // 各線分を処理
  for (size_t i = 1; i < sorted_lines.size(); i++) {
    float angle_diff = std::abs(sorted_lines[i].angle - current_angle);

    // 角度が180度近くの場合の特殊処理（例：179度と1度は実質的に同じ方向）
    if (angle_diff > 180.0f - angle_tolerance_deg) {
      angle_diff = 180.0f - angle_diff;
    }

    // 角度差がトレランスを超える場合は新しいクラスターを作成
    if (angle_diff > angle_tolerance_deg) {
      current_cluster++;
      current_angle = sorted_lines[i].angle;
    }

    clusters[current_cluster].push_back(sorted_lines[i]);
  }

  // クラスター内の線分を元の順序に戻す（オプション）
  for (auto & cluster : clusters) {
    std::sort(
      cluster.second.begin(), cluster.second.end(),
      [](const LineInfo & a, const LineInfo & b) {return a.index < b.index;});
  }

  return clusters;
}

std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> extractRectangleLines(
  const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> & line_segments,
  float angle_tolerance = 15.0f,
  float min_rectangle_score = 0.7f)
{
  if (line_segments.size() < 4) {
    std::cerr << "rectangle detection: too few line segments: " << line_segments.size() <<
      std::endl;
    return {};    // 少なくとも4つの線分が必要
  }

  std::vector<LineInfo> line_infos;
  line_infos.reserve(line_segments.size());

  for (size_t i = 0; i < line_segments.size(); ++i) {
    const auto & line = line_segments[i];
    Eigen::Vector2f dir = line.second - line.first;
    float length = dir.norm();

    if (length < 1e-5f) {
      continue;                      // 極小の線分は無視

    }
    Eigen::Vector2f normalized_dir = dir / length;
    float angle = std::atan2(normalized_dir.y(), normalized_dir.x()) * 180.0f / M_PI;
    if (angle < 0) {
      // angle += 180.0f;                 // 0〜180度の範囲に正規化
      angle = std::fabs(angle);

    }
    line_infos.push_back({normalized_dir, length, angle, static_cast<int>(i)});
  }

  // 線分をおおよその方向でグループ化 (平行線を見つける)
  std::map<int, std::vector<LineInfo>> angle_groups = clusterLinesByAngle(
    line_infos,
    angle_tolerance);

  // 2つの主な方向を見つける (長方形には2つの主な方向がある)
  std::vector<std::pair<int, std::vector<LineInfo>>> major_direction_groups;

  // 各グループの合計線分長を計算して、大きい順にソート
  for (const auto & group : angle_groups) {
    float total_length = 0;
    for (const auto & line : group.second) {
      total_length += line.length;
    }

    major_direction_groups.push_back({group.first, group.second});
  }

  // 合計長さでソート (降順)
  std::sort(
    major_direction_groups.begin(), major_direction_groups.end(),
    [](const auto & a, const auto & b) {
      float len_a = std::accumulate(
        a.second.begin(), a.second.end(), 0.0f,
        [](float sum, const LineInfo & line) {return sum + line.length;});
      float len_b = std::accumulate(
        b.second.begin(), b.second.end(), 0.0f,
        [](float sum, const LineInfo & line) {return sum + line.length;});
      return len_a > len_b;
    });

  // 少なくとも2つの主要な方向が必要
  if (major_direction_groups.size() < 2) {
    std::cerr << "rectangle detection: too few major directions: " <<
      major_direction_groups.size() <<
      std::endl;
    return {};
  }

  // 2つの主要な方向が約90度を成すかチェック
  const auto & group1 = major_direction_groups[0].second;
  const auto & group2 = major_direction_groups[1].second;

  float angle1 = group1[0].angle;
  float angle2 = group2[0].angle;

  float angle_diff = std::abs(angle1 - angle2);
  if (std::abs(angle_diff - 90.0f) > angle_tolerance &&
    std::abs(angle_diff - 90.0f) < 180.0f - angle_tolerance)
  {
    // 方向が90度±許容誤差でない場合
    std::cerr << "rectangle detection: major directions are not orthogonal: " <<
      angle_diff << " degrees" << std::endl;
    return {};
  }

  // 各方向から最長の2本を選ぶ
  auto select_longest_two = [&line_segments](const std::vector<LineInfo> & lines) -> std::vector<std::pair<Eigen::Vector2f,
      Eigen::Vector2f>> {
      std::vector<LineInfo> sorted_lines = lines;
      std::sort(
        sorted_lines.begin(), sorted_lines.end(),
        [](const LineInfo & a, const LineInfo & b) {return a.length > b.length;});

      std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> result;
      size_t count = std::min(size_t(2), sorted_lines.size());
      for (size_t i = 0; i < count; ++i) {
        result.push_back(line_segments[sorted_lines[i].index]);
      }
      return result;
    };

  auto longest_lines1 = select_longest_two(group1);
  auto longest_lines2 = select_longest_two(group2);

  // 結果を結合
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> rectangle_lines;
  rectangle_lines.insert(rectangle_lines.end(), longest_lines1.begin(), longest_lines1.end());
  rectangle_lines.insert(rectangle_lines.end(), longest_lines2.begin(), longest_lines2.end());

  // 線分が4本未満の場合、残りを補完
  while (rectangle_lines.size() < 4) {
    if (longest_lines1.size() > rectangle_lines.size() / 2) {
      // 方向1からもう1本追加
      rectangle_lines.push_back(line_segments[group1[rectangle_lines.size() / 2].index]);
    } else if (longest_lines2.size() > (rectangle_lines.size() - longest_lines1.size())) {
      // 方向2からもう1本追加
      rectangle_lines.push_back(
        line_segments[group2[rectangle_lines.size() -
        longest_lines1.size()].index]);
    } else {
      break;     // これ以上追加できない
    }
  }

  return rectangle_lines;
}

// 使用例
cv::RotatedRect detectRectangle(cv::Mat & image, const std::vector<cv::Vec4i> & cv_lines)
{
  // OpenCV線分形式からEigen形式に変換
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> line_segments;
  for (const auto & l : cv_lines) {
    line_segments.push_back(
    {
      Eigen::Vector2f(l[0], l[1]),
      Eigen::Vector2f(l[2], l[3])
    });
  }

  // 長方形を構成する線分を抽出
  auto rectangle_lines = extractRectangleLines(line_segments);
  std::cout << "Rectangle Lines: " << rectangle_lines.size() << std::endl;

  // 結果の描画
  cv::Mat rectangle_image = image.clone();

  // 全ての検出線を薄い灰色で描画
  for (const auto & l : cv_lines) {
    cv::line(
      rectangle_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
      cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
  }

  // 抽出された長方形の線分を赤色で強調表示
  for (const auto & line : rectangle_lines) {
    cv::Point pt1(line.first.x(), line.first.y());
    cv::Point pt2(line.second.x(), line.second.y());
    std::cout << pt1 << " -> " << pt2 << std::endl;
    cv::line(rectangle_image, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  }

  cv::imshow("Rectangle Detection2", rectangle_image);
  // cv::imwrite("rectangle_detection.jpg", rectangle_image);
  cv::RotatedRect rotatedRect = calculateRotatedRectFromLines(rectangle_lines);
  drawRotatedRect(rectangle_image, rotatedRect, cv::Scalar(0, 255, 0), 2);
  cv::imshow("RESULT: ", rectangle_image);
  return rotatedRect;
}


int main(int argc, char * argv[])
{
  #if 0
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    point_cloud->push_back(pcl::PointXYZ(16, 32, 4));
    dklib::perception::image_processing::IteratableColorizedPointCloudReadOnlyAccessor<pcl::PointCloud<pcl::PointXYZ>>
    accessor(*point_cloud);
    std::cout << "pcl::PointXYZ ---" << std::endl;
    std::cout << accessor << std::endl;
  }

  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    point_cloud->push_back(pcl::PointXYZRGB(16, 32, 4, 100, 10, 4));
    dklib::perception::image_processing::IteratableColorizedPointCloudReadOnlyAccessor<pcl::PointCloud<pcl::PointXYZRGB>>
    accessor(*point_cloud);
    std::cout << "pcl::PointXYZRGB ---" << std::endl;
    std::cout << accessor << std::endl;
  }
#endif

  {
    cv::Mat rgb_img = cv::imread(
      "/home/dean/workspace/src/dklib/.archives/scripts/rgbd_data/color.jpg");
    cv::Mat depth_img = cv::imread(
      "/home/dean/workspace/src/dklib/.archives/scripts/rgbd_data/depth.png", cv::IMREAD_ANYDEPTH);
    std::string camera_info_file =
      "/home/dean/workspace/src/dklib/.archives/scripts/rgbd_data/rgb_camera_info.json";
    Eigen::Matrix3f intrinsic;
    intrinsic << 419.82196044921875, 0.0, 420.063720703125, 0.0, 418.7557067871094,
      246.293212890625, 0.0,
      0.0, 1.0;
    dklib::perception::type::pointcloud::DepthImageSet rgbd_image_set(rgb_img, depth_img, intrinsic,
      0.001);


    const float meter_per_pixel = 0.005;
    Eigen::Vector3f translation_to_plane_origin(0, 0, 0);
    Eigen::Quaternionf rotation_to_plane{Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX())};

    dklib::perception::image_processing::PointCloudToOrthoImageProjector
      projector(translation_to_plane_origin, rotation_to_plane);

    cv::Size img_size = cv::Size{500, 500};
    projector.setProjectedImageSize(img_size);
    dklib::perception::type::pointcloud::IteratableColorizedPointCloudReadOnlyAccessor<
      dklib::perception::type::pointcloud::DepthImageSet> accessor(rgbd_image_set);

    cv::Mat img = projector.project(accessor, meter_per_pixel, std::greater<float>());
    // bgr to gray, and threshold
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY);
    cv::imshow("gray", gray);
    cv::imshow("bin", bin);

    {
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
      cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);
      cv::Mat labels, stats, centroids;
      int nLabels = cv::connectedComponentsWithStats(bin, labels, stats, centroids);

      cv::Point image_center = cv::Point(img.cols / 2, img.rows / 2);
      auto distances = calculateAverageDistanceFromPoint(labels, image_center, nLabels);
      int nearest_label = -1;
      double nearest_distance = std::numeric_limits<double>::max();
      for (size_t i = 1; i < distances.size(); i++) {
        std::cout << "Label " << i << ": " << distances[i] << std::endl;
        if (distances[i] < nearest_distance) {
          nearest_label = i;
          nearest_distance = distances[i];
        }
      }

      std::cout << "nearest label: " << nearest_label << std::endl;


      std::cout << labels.size() << std::endl;
      cv::Mat labeledImage = cv::Mat::zeros(bin.size(), CV_8UC3);
      std::vector<cv::Vec3b> colors(nLabels);
      colors[0] = cv::Vec3b(0, 0, 0); // Background color (black)
      for (int i = 1; i < nLabels; ++i) {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
      }

      for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
          int label = labels.at<int>(y, x);
          labeledImage.at<cv::Vec3b>(y, x) = colors[label];
        }
      }


      cv::Mat selected_label_image = cv::Mat::zeros(bin.size(), CV_8UC1);
      for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
          int label = labels.at<int>(y, x);
          if (label != nearest_label) {continue;}

          selected_label_image.at<uint8_t>(y, x) = 255;
        }
      }


      cv::imshow("Labeled Image", labeledImage);
      cv::imshow("Selected Labeled Image", selected_label_image);
      cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 50));
      cv::morphologyEx(selected_label_image, selected_label_image, cv::MORPH_CLOSE, kernel2);
      cv::imshow("Selected Labeled Image(EX)", selected_label_image);

      cv::Mat canny;
      cv::Canny(selected_label_image, canny, 50, 150);
      cv::imshow("canny", canny);
      std::vector<cv::Vec4i> lines;
      cv::HoughLinesP(canny, lines, 1, CV_PI / 180, 50, 20, 25);

      cv::Mat selected_label_image_color;
      cv::cvtColor(selected_label_image, selected_label_image_color, cv::COLOR_GRAY2BGR);
      cv::RotatedRect rotatedRect = detectRectangle(selected_label_image_color, lines);
      std::cout << "Rotated Rectangle:" << std::endl;
      std::cout << "  Center: " << rotatedRect.center * meter_per_pixel << std::endl;
      std::cout << "  Size: " << rotatedRect.size * meter_per_pixel << std::endl;
      std::cout << "  Angle: " << rotatedRect.angle << std::endl;


      cv::Mat lineImage = cv::Mat::zeros(selected_label_image.size(), CV_8UC3);
      cv::cvtColor(selected_label_image, lineImage, cv::COLOR_GRAY2BGR);

      for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::line(
          lineImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(
            0, 0,
            255), 2,
          cv::LINE_AA);
      }
      cv::imshow("Detected Lines", lineImage);

    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat drawing = cv::Mat::zeros(gray.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
      cv::Scalar color = cv::Scalar(255, 255, 255);
      cv::drawContours(drawing, contours, i, color, 2, cv::LINE_8, hierarchy, 0);
    }
    cv::imshow("contours", drawing);

    cv::imshow("rgb", rgb_img);
    cv::imshow("result", img);
    cv::imwrite("ortho.jpg", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // std::cout << "DepthImageSet ---" << std::endl;
    // std::cout << accessor << std::endl;
  }
  return 0;
}
