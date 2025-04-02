// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "opencv2/ccalib/multicam_calibrator.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"
#include "opencv2/ccalib/fiducial_detectors.hpp"
#include "opencv2/ccalib/visualizer.hpp"
#include "precomp.hpp"
#include <iostream>
#include <map>
#include <sys/stat.h>

namespace {
// Global constant for the default dictionary.
const cv::aruco::PredefinedDictionaryType DEFAULT_DICT = cv::aruco::DICT_4X4_50;

// Helper function to compute the relative rotation and translation from the
// reference camera.
void computeRelativePose(const cv::Mat &rRef, const cv::Mat &tRef,
                         const cv::Mat &rCam, const cv::Mat &tCam,
                         cv::Mat &R_rel, cv::Mat &t_rel) {
  cv::Mat R_ref, R_cam;
  cv::Rodrigues(rRef, R_ref);
  cv::Rodrigues(rCam, R_cam);
  R_rel = R_cam * R_ref.t();
  t_rel = tCam - R_rel * tRef;
}

// Helper: Map a dictionary name string to a predefined ArUco dictionary.
cv::Ptr<cv::aruco::Dictionary> getArucoDictionary(const std::string &dictName) {
  std::map<std::string, cv::aruco::PredefinedDictionaryType> dictMap = {
      {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
      {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
      {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
      {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
      {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
      {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
      {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
      {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
      {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
      {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
      {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
      {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
      {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
      {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
      {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
      {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
      {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
      {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
      {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
      {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
      {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}};

  auto it = dictMap.find(dictName);
  if (it != dictMap.end()) {
    return cv::makePtr<cv::aruco::Dictionary>(
        cv::aruco::getPredefinedDictionary(it->second));
  }
  // Use the default dictionary if not found.
  return cv::makePtr<cv::aruco::Dictionary>(
      cv::aruco::getPredefinedDictionary(DEFAULT_DICT));
}

// Helper: Load the dictionary based on a provided dictionary string.
cv::Ptr<cv::aruco::Dictionary> loadDictionary(const std::string &dictStr) {
  return getArucoDictionary(dictStr);
}
} // namespace

namespace cv {
namespace ccalib {

MultiCameraCalibrator::MultiCameraCalibrator() {}
MultiCameraCalibrator::~MultiCameraCalibrator() {}

void MultiCameraCalibrator::addCamera(const std::string &configPath) {
  CameraData data;
  loadConfig(configPath, data);
  if (data.cameraId < 0) {
    std::cerr << "Invalid camera_id in " << configPath << std::endl;
    return;
  }
  if (referenceCameraId_ < 0) {
    referenceCameraId_ = data.cameraId;
  }
  cameraDatas_[data.cameraId] = data;
}

void MultiCameraCalibrator::loadMultiCameraConfig(
    const std::string &configPath) {
  cv::FileStorage fs(configPath, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Cannot open multi-camera config: " << configPath << std::endl;
    return;
  }
  cv::FileNode camerasNode = fs["cameras"];
  if (camerasNode.type() != cv::FileNode::MAP) {
    std::cerr << "Invalid format: 'cameras' node is not a map." << std::endl;
    return;
  }
  for (cv::FileNodeIterator it = camerasNode.begin(); it != camerasNode.end();
       ++it) {
    CameraData data;
    loadConfigFromNode(*it, data);
    if (data.cameraId < 0) {
      std::cerr << "Invalid camera_id in one of the config nodes." << std::endl;
      continue;
    }
    if (referenceCameraId_ < 0) {
      referenceCameraId_ = data.cameraId;
    }
    cameraDatas_[data.cameraId] = data;
  }
  fs.release();
}

void MultiCameraCalibrator::loadConfigFromNode(const cv::FileNode &node,
                                               CameraData &data) {
  data.cameraId = (int)node["camera_id"];
  data.patternType = (std::string)node["pattern_type"];

  // Validate and load pattern_size (must be a sequence of 2 numbers)
  cv::FileNode ps = node["pattern_size"];
  if (ps.type() == cv::FileNode::SEQ && ps.size() == 2) {
    data.patternSize.width = (int)ps[0];
    data.patternSize.height = (int)ps[1];
  } else {
    std::cerr << "Invalid or missing pattern_size for camera " << data.cameraId
              << std::endl;
    data.cameraId = -1;
    return;
  }

  data.squareSize = (float)node["square_size"];
  data.markerSize = (float)node["marker_size"];
  data.dictionary = (std::string)node["dictionary"];
  // Optional calibration flags; default is 0.
  data.calibFlags = (int)node["calib_flags"];

  cv::FileNode ipaths = node["image_paths"];
  if (ipaths.type() == cv::FileNode::SEQ) {
    for (cv::FileNodeIterator it = ipaths.begin(); it != ipaths.end(); ++it) {
      data.imagePaths.push_back((std::string)*it);
    }
  }
}

void MultiCameraCalibrator::loadConfig(const std::string &path,
                                       CameraData &data) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Cannot open config: " << path << std::endl;
    return;
  }
  // Assuming the file itself is a map for one camera.
  loadConfigFromNode(fs.root(), data);
  fs.release();
}

void MultiCameraCalibrator::detectAllPatterns() {
  for (auto &kv : cameraDatas_) {
    auto &cData = kv.second;
    cv::Ptr<FiducialDetector> detector;
    if (cData.patternType == "chessboard") {
      detector =
          cv::makePtr<ChessboardDetector>(cData.patternSize, cData.squareSize);
    } else if (cData.patternType == "circles") {
      detector = cv::makePtr<CircleGridDetector>(cData.patternSize,
                                                 cData.squareSize, false);
    } else if (cData.patternType == "acircles") {
      detector = cv::makePtr<CircleGridDetector>(cData.patternSize,
                                                 cData.squareSize, true);
    } else if (cData.patternType == "aruco") {
      cv::Ptr<cv::aruco::Dictionary> dictPtr = loadDictionary(cData.dictionary);
      detector = cv::makePtr<ArucoDetector>(dictPtr, cData.markerSize,
                                            cv::aruco::DetectorParameters());
    } else if (cData.patternType == "charuco") {
      cv::Ptr<cv::aruco::Dictionary> dictPtr = loadDictionary(cData.dictionary);
      cv::aruco::CharucoBoard board(cData.patternSize, cData.squareSize,
                                    cData.markerSize, *dictPtr);
      detector = cv::makePtr<CharucoDetector>(board);
    } else {
      std::cerr << "Unknown pattern type: " << cData.patternType << std::endl;
      continue;
    }

    for (const auto &imgPath : cData.imagePaths) {
      cv::Mat img = cv::imread(imgPath);
      if (img.empty()) {
        std::cerr << "Cannot read: " << imgPath << std::endl;
        continue;
      }
      std::vector<cv::Point3f> objPts;
      std::vector<cv::Point2f> imgPts;
      bool ok = detector->detect(img, objPts, imgPts);
      if (ok && !imgPts.empty()) {
        cData.allObjPoints.push_back(objPts);
        cData.allImgPoints.push_back(imgPts);
        cData.imageSize = img.size();

        if (display_) {
          cv::Mat displayImg = img.clone();
          for (size_t i = 0; i < imgPts.size(); i++) {
            cv::circle(displayImg, imgPts[i], 4, cv::Scalar(0, 0, 255), -1);
          }
          cv::imshow("Detections", displayImg);
          int key = cv::waitKey(300);
          if (key == 27)
            break;
        }
      } else {
        std::cerr << "No detections in " << imgPath << std::endl;
      }
    }
  }
  if (display_) {
    cv::destroyWindow("Detections");
  }
}

void MultiCameraCalibrator::calibrateIntrinsics() {
  for (auto &kv : cameraDatas_) {
    int cid = kv.first;
    auto &cData = kv.second;

    if (cData.allObjPoints.empty()) {
      std::cerr << "No detections for camera " << cid << std::endl;
      continue;
    }

    CameraCalibrationResult res;
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    double rms = cv::calibrateCamera(cData.allObjPoints, cData.allImgPoints,
                                     cData.imageSize, K, dist, rvecs, tvecs,
                                     cData.calibFlags);

    res.cameraMatrix = K.clone();
    res.distCoeffs = dist.clone();
    res.rvecs = rvecs;
    res.tvecs = tvecs;
    res.reprojectionError = rms;
    res.imageSize = cData.imageSize;
    cameraCalib_[cid] = res;

    std::cout << "[Camera " << cid << "] RMS=" << rms << std::endl;
  }
}

void MultiCameraCalibrator::computeRelativePoses() {
  if (cameraCalib_.find(referenceCameraId_) == cameraCalib_.end()) {
    std::cerr << "Reference camera not found in calibration results\n";
    return;
  }
  const auto &refRes = cameraCalib_.at(referenceCameraId_);
  if (refRes.rvecs.empty())
    return;

  cv::Mat rRef = refRes.rvecs[0], tRef = refRes.tvecs[0];
  realPoses_[referenceCameraId_] = {cv::Mat::eye(3, 3, CV_64F),
                                    cv::Mat::zeros(3, 1, CV_64F)};

  for (const auto &kv : cameraCalib_) {
    int cid = kv.first;
    if (cid == referenceCameraId_)
      continue;
    const auto &cRes = kv.second;
    if (cRes.rvecs.empty())
      continue;

    cv::Mat R_rel, t_rel;
    computeRelativePose(rRef, tRef, cRes.rvecs[0], cRes.tvecs[0], R_rel, t_rel);
    realPoses_[cid] = {R_rel, t_rel};
  }
}

void MultiCameraCalibrator::refineWithBundleAdjustment() {
  // This is a future implementation placeholder, I will implement it later.
  // Bundle adjustment is a complex optimization problem that requires
  // a lot of data and careful handling of the optimization process.
  // For now, I will just print a message.
  std::cout << "[INFO] refineWithBundleAdjustment() not implemented.\n";
}

void MultiCameraCalibrator::calibrate(bool computeRelative) {
  detectAllPatterns();
  calibrateIntrinsics();

  if (computeRelative) {
    computeRelativePoses();
  }

  // ------------------------------------------------------------------
  //  This is the code for producing the error plots for each camera
  // ------------------------------------------------------------------
  CalibrationVisualizer viz;

  for (const auto &kv : cameraCalib_) {
    int cid = kv.first;
    const auto &calRes = kv.second;

    if (calRes.rvecs.empty() || calRes.tvecs.empty())
      continue;

    // Gather the per-frame errors
    auto &cData = cameraDatas_.at(cid); // the stored object/image points
    std::vector<double> perFrameErrors = computePerFrameErrors(
        cData.allObjPoints, cData.allImgPoints, calRes.cameraMatrix,
        calRes.distCoeffs, calRes.rvecs, calRes.tvecs);

    // 1) Make a histogram image
    cv::Mat histImg = viz.plotErrorHistogram(perFrameErrors, 15, 800, 600);

    if (display_) {
      std::string wname =
          " Histogram for Error Distribution of Camera" + std::to_string(cid);
      cv::namedWindow(wname, cv::WINDOW_NORMAL);
      cv::resizeWindow(wname, 800, 600);
      cv::imshow(wname, histImg);
    }

    // 2) Per-frame error line plot
    cv::Mat linePlot = viz.plotReprojErrorsLine(perFrameErrors, 800, 600);

    if (display_) {
      std::string wname2 =
          "Reprojection Errors for camera" + std::to_string(cid);
      cv::namedWindow(wname2, cv::WINDOW_NORMAL);
      cv::resizeWindow(wname2, 800, 600);
      cv::imshow(wname2, linePlot);
    }
  }

  // 3) Show 3D extrinsics
  if (cameraCalib_.size() > 1) {
    viz.plotExtrinsics3D(cameraCalib_, referenceCameraId_, realPoses_);
  }

  if (display_) {
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}

void MultiCameraCalibrator::generateReport(const std::string &outputDir) const {
  // Print text-based summary to the console.
  std::cout << "\n===== Multi-Camera Calibration Report =====\n";
  std::cout << "Reference camera: " << referenceCameraId_ << "\n";
  for (const auto &kv : cameraCalib_) {
    std::cout << "Camera " << kv.first
              << " -> ReprojErr=" << kv.second.reprojectionError << "\n";
  }
  std::cout << "===========================================\n\n";

  // Create and save a simple image summary.
  cv::Mat summary(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::putText(summary, "Calibration Summary", cv::Point(50, 200),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);

  // Create a subfolder "plot" inside outputDir using a system call.
  std::string plotFolder = outputDir + "/plot";
#if defined(_WIN32)
  std::string mkdirCommand = "mkdir " + plotFolder;
#else
  std::string mkdirCommand = "mkdir -p " + plotFolder;
#endif
  system(mkdirCommand.c_str());

  // Create an instance of the CalibrationVisualizer.
  CalibrationVisualizer viz;

  // For each camera, generate and save error plots.
  // (Assuming that cameraDatas_ holds the per-camera data and that
  // computePerFrameErrors(...) is available.)
  for (const auto &kv : cameraCalib_) {
    int cid = kv.first;
    // Check if we have the corresponding camera data.
    auto it = cameraDatas_.find(cid);
    if (it == cameraDatas_.end())
      continue;
    const CameraData &cData = it->second;

    // Compute per-frame errors (this function is assumed available).
    std::vector<double> perFrameErrors = computePerFrameErrors(
        cData.allObjPoints, cData.allImgPoints, kv.second.cameraMatrix,
        kv.second.distCoeffs, kv.second.rvecs, kv.second.tvecs);

    // Generate the histogram image.
    cv::Mat histImg = viz.plotErrorHistogram(perFrameErrors, 15, 800, 600);
    std::string histFilename =
        plotFolder + "/cam" + std::to_string(cid) + "_histogram.png";
    cv::imwrite(histFilename, histImg);

    // Generate the per-frame error line plot.
    cv::Mat linePlot = viz.plotReprojErrorsLine(perFrameErrors, 800, 600);
    std::string lineFilename =
        plotFolder + "/cam" + std::to_string(cid) + "_reprojErrors.png";
    cv::imwrite(lineFilename, linePlot);
  }
}

void MultiCameraCalibrator::saveResults(const std::string &outputPath) const {
  cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "Cannot open file for writing: " << outputPath << std::endl;
    return;
  }
  fs << "reference_camera_id" << referenceCameraId_;

  fs << "cameras" << "[";
  for (const auto &kv : cameraCalib_) {
    int camId = kv.first;
    const auto &cRes = kv.second;
    fs << "{:"
       << "camera_id" << camId << "camera_matrix" << cRes.cameraMatrix
       << "dist_coeffs" << cRes.distCoeffs << "reproj_error"
       << cRes.reprojectionError << "}";
  }
  fs << "]";
  fs << "relative_poses" << "{";
  for (const auto &kv : realPoses_) {
    int cid = kv.first;
    fs << ("cam_" + std::to_string(cid) + "_R") << kv.second.first;
    fs << ("cam_" + std::to_string(cid) + "_t") << kv.second.second;
  }
  fs << "}";
  fs.release();
  std::cout << "Saved multi-camera calibration to " << outputPath << std::endl;
}

void MultiCameraCalibrator::validateCalibration() const {
  std::cout
      << "[ValidateCalibration] Placeholder: implement advanced checks.\n";
}

} // namespace ccalib
} // namespace cv
