// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "opencv2/ccalib/calibration_utils.hpp"
#include "precomp.hpp"
#include <cmath>

namespace cv {
namespace ccalib {

double computeReprojectionError(
    const std::vector<std::vector<cv::Point3f>> &objectPoints,
    const std::vector<std::vector<cv::Point2f>> &imagePoints,
    const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
    const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs) {
  CV_Assert(objectPoints.size() == imagePoints.size());
  CV_Assert(objectPoints.size() == rvecs.size());
  CV_Assert(objectPoints.size() == tvecs.size());

  double totalErr = 0.0;
  size_t totalPoints = 0;
  std::vector<cv::Point2f> projected;

  for (size_t i = 0; i < objectPoints.size(); ++i) {
    cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, projected);

    double err = norm(imagePoints[i], projected, cv::NORM_L2);
    totalErr += err * err;
    totalPoints += objectPoints[i].size();
  }

  return std::sqrt(totalErr / totalPoints);
}

void computeRelativePose(const cv::Mat &rvec1, const cv::Mat &tvec1,
                         const cv::Mat &rvec2, const cv::Mat &tvec2,
                         cv::Mat &R_rel, cv::Mat &t_rel) {
  cv::Mat R1, R2;
  cv::Rodrigues(rvec1, R1);
  cv::Rodrigues(rvec2, R2);

  cv::Mat R1_inv = R1.t();
  R_rel = R2 * R1_inv;
  t_rel = tvec2 - R_rel * tvec1;
}

std::vector<double>
computePerFrameErrors(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                      const std::vector<std::vector<cv::Point2f>> &imagePoints,
                      const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                      const std::vector<cv::Mat> &rvecs,
                      const std::vector<cv::Mat> &tvecs) {
  CV_Assert(objectPoints.size() == imagePoints.size());
  CV_Assert(objectPoints.size() == rvecs.size());
  CV_Assert(objectPoints.size() == tvecs.size());

  std::vector<double> perFrameErrors;
  perFrameErrors.reserve(objectPoints.size());

  std::vector<cv::Point2f> projected;

  // Compute an RMS error for each calibration image.
  for (size_t i = 0; i < objectPoints.size(); ++i) {
    cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, projected);

    double err = norm(imagePoints[i], projected, cv::NORM_L2);
    double rms = std::sqrt((err * err) / objectPoints[i].size());
    perFrameErrors.push_back(rms);
  }
  return perFrameErrors;
}

} // namespace ccalib
} // namespace cv
