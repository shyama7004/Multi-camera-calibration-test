// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP
#define OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

namespace cv {
namespace ccalib {

/**
 * @struct CameraCalibrationResult
 * @brief Holds the final intrinsics and extrinsics for a single camera.
 */
struct CV_EXPORTS CameraCalibrationResult {
  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  double reprojectionError = -1.0;
  cv::Size imageSize;
};

/**
 * @class MultiCameraCalibrator
 * @brief High-level manager for calibrating multiple cameras.
 *
 * This class can load configuration for one or several cameras (via a single
 * YAML/JSON file), detect calibration patterns in images, compute intrinsic and
 * extrinsic calibration, and generate both text and image-based reports.
 */
class CV_EXPORTS MultiCameraCalibrator {
public:
  MultiCameraCalibrator();
  ~MultiCameraCalibrator();

  /**
   * @brief Load a single camera dataset from a config file.
   * (For backward compatibility when using one file per camera.)
   */
  void addCamera(const std::string &configPath);

  /**
   * @brief Load multiple cameras dataset from a single config file.
   *
   * The file should have a “cameras” map with each camera’s config, for
   * example:
   * @code
   * cameras:
   *   cam1:
   *     camera_id: 1
   *     pattern_type: chessboard
   *     pattern_size: [10, 7]
   *     square_size: 0.028
   *     image_paths: ["/path/to/img0.png", ...]
   *   cam2:
   *     camera_id: 2
   *     pattern_type: chessboard
   *     pattern_size: [10, 7]
   *     square_size: 0.028
   *     image_paths: ["/path/to/img0.png", ...]
   * @endcode
   */
  void loadMultiCameraConfig(const std::string &configPath);

  /**
   * @brief Calibrate each camera individually and optionally compute relative
   * extrinsics.
   * @param computeRelative If true, computeRelativePoses() is called using the
   * first camera as reference.
   */
  void calibrate(bool computeRelative = true);

  /**
   * @brief Save intrinsics and extrinsics for all cameras to a single file.
   * @param outputPath path to the output YAML
   */
  void saveResults(const std::string &outputPath) const;

  /**
   * @brief Generate a minimal text-based or image-based analysis report.
   * @param outputDir Directory where the report image will be saved.
   */
  void generateReport(const std::string &outputDir) const;

  /**
   * @brief Validate calibration across cameras by cross-checking reprojected
   * corners, etc.
   */
  void validateCalibration() const;

  /**
   * @brief Enable or disable visualization.
   */
  void setDisplay(bool enable) { display_ = enable; }

private:
  // Internal data structure for each camera.
  struct CameraData {
	int cameraId = -1;
	std::string
		patternType; // e.g., "chessboard", "charuco", "circles", "aruco", etc.
	cv::Size patternSize;
	float squareSize = 0.f; // For chessboard, circle grid, etc.
	float markerSize = 0.f; // For charuco, aruco
	std::string dictionary; // e.g., "DICT_6X6_250"
	int calibFlags = 0;     // Calibration flags (default 0)

	std::vector<std::string> imagePaths;

	// Accumulated calibration points.
	std::vector<std::vector<cv::Point3f>> allObjPoints;
	std::vector<std::vector<cv::Point2f>> allImgPoints;
	cv::Size imageSize;
  };

  // Mapping from camera ID to its calibration data.
  std::map<int, CameraData> cameraDatas_;
  // Mapping from camera ID to its final calibration result.
  std::map<int, CameraCalibrationResult> cameraCalib_;
  // Reference camera ID (first loaded).
  int referenceCameraId_ = -1;
  // Store (R, t) for each non-reference camera relative to the reference.
  std::map<int, std::pair<cv::Mat, cv::Mat>> realPoses_;

  bool display_ = true;

  // Loads a configuration from a file (for a single camera).
  void loadConfig(const std::string &path, CameraData &data);

  // Load config data from FileNode (for single/multi-camera).
  void loadConfigFromNode(const cv::FileNode &node, CameraData &data);

  // Detect calibration patterns in all images.
  void detectAllPatterns();

  // Calibrate intrinsics for each camera.
  void calibrateIntrinsics();

  // Compute camera extrinsics relative to reference.
  void computeRelativePoses();

  // Refine calibration via bundle adjustment (customizable).
  void refineWithBundleAdjustment();
};

} // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP
