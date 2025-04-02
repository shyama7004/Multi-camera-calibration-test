// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CCALIB_CALIBRATION_UTILS_HPP
#define OPENCV_CCALIB_CALIBRATION_UTILS_HPP

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace ccalib {

/**
 * @brief Compute average reprojection error across multiple frames.
 */
double computeReprojectionError(
    const std::vector<std::vector<cv::Point3f>> &objectPoints,
    const std::vector<std::vector<cv::Point2f>> &imagePoints,
    const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
    const std::vector<cv::Mat> &rvecs, const std::vector<cv::Mat> &tvecs);

/**
 * @brief Compute relative pose (R_rel, t_rel) from camera1 to camera2, given
 * their extrinsics in the world frame.
 */
void computeRelativePose(const cv::Mat &rvec1, const cv::Mat &tvec1,
                         const cv::Mat &rvec2, const cv::Mat &tvec2,
                         cv::Mat &R_rel, cv::Mat &t_rel);

/**
 * @brief Compute per-frame RMS reprojection errors. Each returned entry
 * corresponds to the RMS error for a single calibration image.
 * @param objectPoints 3D corners per frame
 * @param imagePoints 2D corners per frame
 * @param cameraMatrix Intrinsics
 * @param distCoeffs Distortion
 * @param rvecs Per-frame rotation
 * @param tvecs Per-frame translation
 * @return A vector of RMS errors, one entry per frame.
 */
std::vector<double>
computePerFrameErrors(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                      const std::vector<std::vector<cv::Point2f>> &imagePoints,
                      const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                      const std::vector<cv::Mat> &rvecs,
                      const std::vector<cv::Mat> &tvecs);

} // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_CALIBRATION_UTILS_HPP
