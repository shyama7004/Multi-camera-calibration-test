// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <iostream>
#include <opencv2/ccalib/fiducial_detectors.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

namespace cv {
namespace ccalib {
// this is a helper function to convert an image to grayscale.
namespace {
cv::Mat toGray(const cv::Mat &image) {
  if (image.channels() > 1) {
    cv::Mat gray;
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  return image;
}
} // namespace

// Build a 3D chessboard object points vector based on board size and square
// size.
static std::vector<cv::Point3f> buildChessboard3D(cv::Size sz, float sqSize) {
  std::vector<cv::Point3f> pts;
  pts.reserve(sz.width * sz.height);
  // Loop over rows and columns to create object points.
  for (int rows = 0; rows < sz.height; ++rows) {
    for (int cols = 0; cols < sz.width; ++cols) {
      pts.emplace_back(cols * sqSize, rows * sqSize, 0.f);
    }
  }
  return pts;
}

// Constructor for ChessboardDetector: initializes parameters and precomputed
// object points.
ChessboardDetector::ChessboardDetector(cv::Size patternSize, float squareSize,
                                       int detectionFlags,
                                       cv::TermCriteria subPixCriteria)
    : patternSize_(patternSize), squareSize_(squareSize),
      detectionFlags_(detectionFlags), subPixCriteria_(subPixCriteria),
      precomputedObjectPoints_(buildChessboard3D(patternSize, squareSize)) {}
// Set the detection flags for the chessboard detector.
void ChessboardDetector::setDetectionFlags(int flags) {
  detectionFlags_ = flags;
}

// Set the criteria for sub-pixel corner refinement.
void ChessboardDetector::setSubPixCriteria(cv::TermCriteria criteria) {
  subPixCriteria_ = criteria;
}

// Detect chessboard corners in the input image.
bool ChessboardDetector::detect(cv::InputArray inImage,
                                std::vector<cv::Point3f> &objectPoints,
                                std::vector<cv::Point2f> &imagePoints) {
  cv::Mat image = inImage.getMat();
  if (image.empty()) {
    CV_LOG_WARNING(NULL, "Input image is empty");
    return false;
  }

  cv::Mat grayImage = toGray(image);

  bool found = findChessboardCorners(grayImage, patternSize_, imagePoints,
                                     detectionFlags_);

  if (!found) {
    CV_LOG_WARNING(NULL, "Chessboard corners not found");
    return false;
  }

  // Refine corner positions using sub-pixel accuracy.
  cornerSubPix(grayImage, imagePoints, cv::Size(11, 11), cv::Size(-1, -1),
               subPixCriteria_);
  objectPoints = precomputedObjectPoints_;
  return true;
}

// Build a 3D circle grid object points vector based on grid size, circle
// distance, and asymmetry flag.
static std::vector<cv::Point3f> buildCircle3D(cv::Size sz, float d, bool asym) {
  std::vector<cv::Point3f> pts;
  pts.reserve(sz.width * sz.height);
  // Loop over rows and columns, you can adjust offset for asymmetric grid.
  for (int rows = 0; rows < sz.height; ++rows) {
    for (int cols = 0; cols < sz.width; ++cols) {
      float offset = (asym && (rows % 2 == 1)) ? d * 0.5f : 0.f;
      pts.emplace_back(cols * d + offset, rows * d, 0.f);
    }
  }
  return pts;
}

// Constructor for CircleGridDetector: initializes parameters and precomputed
// object points.
CircleGridDetector::CircleGridDetector(cv::Size patternSize, float circleSize,
                                       bool asymmetric)
    : patternSize_(patternSize), circleSize_(circleSize),
      asymmetric_(asymmetric), precomputedObjectPoints_(buildCircle3D(
                                   patternSize, circleSize, asymmetric)) {}

// Detect circle grid corners in the input image.
bool CircleGridDetector::detect(cv::InputArray inImage,
                                std::vector<cv::Point3f> &objectPoints,
                                std::vector<cv::Point2f> &imagePoints) {
  cv::Mat image = inImage.getMat();
  if (image.empty()) {
    CV_LOG_WARNING(NULL, "Input image is empty");
    return false;
  }

  // Use helper function for grayscale conversion.
  cv::Mat grayImage = toGray(image);

  // Set appropriate flags based on whether grid is asymmetric.
  int flags = asymmetric_
                  ? (cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING)
                  : cv::CALIB_CB_SYMMETRIC_GRID;

  bool found = findCirclesGrid(grayImage, patternSize_, imagePoints, flags);
  if (!found) {
    CV_LOG_WARNING(NULL, "Circle grid corners not found");
    return false;
  }

  objectPoints = precomputedObjectPoints_;
  return true;
}

// Constructor for CharucoDetector: initializes the underlying ArUco Charuco
// detector.
CharucoDetector::CharucoDetector(
    const cv::aruco::CharucoBoard &board,
    const cv::aruco::CharucoParameters &charucoParams,
    const cv::aruco::DetectorParameters &detectorParams,
    const cv::aruco::RefineParameters &refineParams) {
  detectorImpl_ = cv::makePtr<cv::aruco::CharucoDetector>(
      board, charucoParams, detectorParams, refineParams);
}

// Detect Charuco board in the input image.
bool CharucoDetector::detect(cv::InputArray inImage,
                             std::vector<cv::Point3f> &objectPoints,
                             std::vector<cv::Point2f> &imagePoints) {
  cv::Mat image = inImage.getMat();

  if (image.empty()) {
    CV_LOG_WARNING(NULL, "Input image is empty");
    return false;
  }
  cv::Mat grayImage = toGray(image);

  std::vector<cv::Point2f> charucoCorners;
  std::vector<int> charucoIds;

  // Detect Charuco board corners and corresponding IDs.
  detectorImpl_->detectBoard(grayImage, charucoCorners, charucoIds);

  // Verify that corners and IDs are detected.
  if (charucoCorners.empty() || charucoIds.empty()) {
    CV_LOG_WARNING(NULL, "Charuco corners or IDs not found");
    return false;
  }

  detectorImpl_->getBoard().matchImagePoints(charucoCorners, charucoIds,
                                             objectPoints, imagePoints);
  if (objectPoints.empty() || imagePoints.empty()) {
    CV_LOG_WARNING(NULL, "No matching points found");
    return false;
  }

  return true;
}

// Get the associated Charuco board.
const cv::aruco::CharucoBoard &CharucoDetector::getBoard() const {
  return detectorImpl_->getBoard();
}

// Constructor for ArucoDetector: initializes dictionary, marker length, and
// detector parameters.
ArucoDetector::ArucoDetector(
    const cv::Ptr<cv::aruco::Dictionary> &dictionary, float markerLength,
    const cv::aruco::DetectorParameters &detectorParams)
    : dictionary_(dictionary), markerLength_(markerLength),
      detectorParams_(detectorParams) {}

// Build a 3D marker object points vector based on marker length.
static std::vector<cv::Point3f> buildMarker3D(float markerLength) {
  return {{0.f, 0.f, 0.f},
          {markerLength, 0.f, 0.f},
          {markerLength, markerLength, 0.f},
          {0.f, markerLength, 0.f}};
}

// Detect ArUco markers in the input image.
bool ArucoDetector::detect(cv::InputArray inImage,
                           std::vector<cv::Point3f> &objectPoints,
                           std::vector<cv::Point2f> &imagePoints) {
  cv::Mat image = inImage.getMat();
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

  cv::aruco::ArucoDetector detector(*dictionary_, detectorParams_);
  detector.detectMarkers(image, markerCorners, markerIds, rejectedCandidates);

  if (markerIds.empty())
    return false;

  objectPoints.clear();
  imagePoints.clear();

  std::vector<cv::Point3f> markerObjPts = buildMarker3D(markerLength_);

  for (size_t i = 0; i < markerIds.size(); i++) {
    objectPoints.insert(objectPoints.end(), markerObjPts.begin(),
                        markerObjPts.end());
    imagePoints.insert(imagePoints.end(), markerCorners[i].begin(),
                       markerCorners[i].end());
  }

  return true;
}

// Get the dictionary used for marker detection.
const cv::Ptr<cv::aruco::Dictionary> &ArucoDetector::getDictionary() const {
  return dictionary_;
}

// Get the marker length parameter.
float ArucoDetector::getMarkerLength() const { return markerLength_; }

} // namespace ccalib
} // namespace cv
