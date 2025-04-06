// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CCALIB_VISUALIZER_HPP
#define OPENCV_CCALIB_VISUALIZER_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/multicam_calibrator.hpp>
#include <opencv2/core.hpp>

namespace cv {
namespace ccalib {

// Color constants for consistency across visualizations.
namespace vizcolors {
    const cv::Scalar PLOT_BG(255, 255, 255);
    const cv::Scalar AXIS_COLOR(0, 0, 0);
    const cv::Scalar ERROR_COLOR(255, 0, 0);
    const cv::Scalar DETECTED_COLOR(0, 255, 0);
    // The following color is used for reprojected corners (yellowish).
    const cv::Scalar REPROJECTED_COLOR(226, 211, 5);
}

class CV_EXPORTS CalibrationVisualizer {
public:
  CalibrationVisualizer();

  /**
   * @brief Draw reprojection error vectors on top of an image.
   */
  cv::Mat drawReprojErrorMap(const cv::Mat &image,
                             const std::vector<cv::Point2f> &detectedCorners,
                             const std::vector<cv::Point2f> &reprojectedCorners) const;

  /**
   * @brief Visualize multi-camera extrinsics in 3D (requires opencv_viz).
   */
  void plotExtrinsics3D(
      const std::map<int, CameraCalibrationResult> &calibResults,
      int referenceCameraId,
      const std::map<int, std::pair<cv::Mat, cv::Mat>> &realPoses,
      const std::string &windowName = "MultiCamera Extrinsics") const;

  /**
   * @brief Produce an image showing how the lens warps a synthetic grid.
   * Note: File saving is removed to separate visualization from I/O.
   */
  cv::Mat drawDistortionGrid(const CameraCalibrationResult &res, int gridSize,
                             cv::Size imageSize) const;

  /**
   * @brief A histogram of errors with axis labels for “Reprojection Error (X - axis)”
   * vs. “Frequency (Y - axis)”.
   */
  cv::Mat plotErrorHistogram(const std::vector<double> &errors,
                             int histSize = 30, int histWidth = 400,
                             int histHeight = 300) const;

  /**
   * @brief A simple line plot for per-frame errors (x-axis = frame index, y-axis = error).
   */
  cv::Mat plotReprojErrorsLine(const std::vector<double> &errors,
                               int width = 400, int height = 300) const;
};

} // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_VISUALIZER_HPP
