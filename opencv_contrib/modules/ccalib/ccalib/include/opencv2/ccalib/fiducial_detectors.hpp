// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CCALIB_FIDUCIAL_DETECTORS_HPP
#define OPENCV_CCALIB_FIDUCIAL_DETECTORS_HPP

#include "opencv2/core.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

namespace cv {
namespace ccalib {
/**
 * @brief Base class for fiducial detectors.
 */
class CV_EXPORTS FiducialDetector {
public:
	virtual ~FiducialDetector() {}
	/**
	 * @brief Detect fiducial points in the given image.
	 * @param image Input image.
	 * @param objectPoints Output 3D object points.
	 * @param imagePoints Output 2D image points.
	 * @return true if detection was successful.
	 */
	virtual bool detect(cv::InputArray image,
											std::vector<cv::Point3f> &objectPoints,
											std::vector<cv::Point2f> &imagePoints) = 0;
};

/**
 * @brief Detector for chessboard patterns.
 */
class CV_EXPORTS ChessboardDetector : public FiducialDetector {
public:
	/**
	 * @brief Constructs a ChessboardDetector.
	 * @param patternSize Number of inner corners per chessboard row and column.
	 * @param squareSize Size of a square in user-defined units.
	 * @param detectionFlags Flags for corner detection.
	 * @param subPixCriteria Termination criteria for corner refinement.
	 */
	ChessboardDetector(cv::Size patternSize, float squareSize,
										 int detectionFlags = cv::CALIB_CB_ADAPTIVE_THRESH |
																					cv::CALIB_CB_NORMALIZE_IMAGE |
																					cv::CALIB_CB_FAST_CHECK,
										 cv::TermCriteria subPixCriteria = cv::TermCriteria(
												 cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
												 0.001));

	bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints,
							std::vector<cv::Point2f> &imagePoints) override;

	/**
	 * @brief Set new detection flags.
	 * @param flags New flags.
	 */
	void setDetectionFlags(int flags);
	/**
	 * @brief Set new subpixel refinement criteria.
	 * @param criteria New termination criteria.
	 */
	void setSubPixCriteria(cv::TermCriteria criteria);
	float getSquareSize() const { return squareSize_; } // maybe needed later

private:
	cv::Size patternSize_;
	float squareSize_;
	int detectionFlags_;
	cv::TermCriteria subPixCriteria_;
	std::vector<cv::Point3f> precomputedObjectPoints_;
};

/**
 * @brief Detector for Aruco markers.
 */
class CV_EXPORTS ArucoDetector : public FiducialDetector {
public:
	/**
	 * @brief Constructs an ArucoDetector.
	 * @param dictionary Aruco dictionary.
	 * @param markerLength Length of the marker's side.
	 * @param detectorParams Detector parameters.
	 */
	ArucoDetector(const cv::Ptr<cv::aruco::Dictionary> &dictionary,
								float markerLength,
								const cv::aruco::DetectorParameters &detectorParams =
										cv::aruco::DetectorParameters());

	bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints,
							std::vector<cv::Point2f> &imagePoints) override;

	/**
	 * @brief Returns the associated Aruco dictionary.
	 * @return Reference to the dictionary.
	 */
	const cv::Ptr<cv::aruco::Dictionary> &getDictionary() const;
	/**
	 * @brief Returns the marker length.
	 * @return Marker length.
	 */
	float getMarkerLength() const;

private:
	cv::Ptr<cv::aruco::Dictionary> dictionary_;
	float markerLength_;
	cv::aruco::DetectorParameters detectorParams_;
};

/**
 * @brief Detector for circle grid patterns.
 */
class CV_EXPORTS CircleGridDetector : public FiducialDetector {
public:
	/**
	 * @brief Constructs a CircleGridDetector.
	 * @param patternSize Number of circles per row and column.
	 * @param circleSize Diameter of a circle.
	 * @param asymmetric Flag indicating whether the grid is asymmetric.
	 */
	CircleGridDetector(cv::Size patternSize, float circleSize, bool asymmetric);

	bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints,
							std::vector<cv::Point2f> &imagePoints) override;

	float getCircleSize() const { return circleSize_; } // maybe needed later

private:
	cv::Size patternSize_;
	float circleSize_;
	bool asymmetric_;
	std::vector<cv::Point3f> precomputedObjectPoints_;
};

/**
 * @brief Detector for ChArUco boards.
 */
class CV_EXPORTS CharucoDetector : public FiducialDetector {
public:
	/**
	 * @brief Constructs a CharucoDetector.
	 * @param board The ChArUco board.
	 * @param charucoParams Parameters specific to ChArUco detection.
	 * @param detectorParams General Aruco detector parameters.
	 * @param refineParams Parameters for refining detections.
	 */
	CharucoDetector(const cv::aruco::CharucoBoard &board,
									const cv::aruco::CharucoParameters &charucoParams =
											cv::aruco::CharucoParameters(),
									const cv::aruco::DetectorParameters &detectorParams =
											cv::aruco::DetectorParameters(),
									const cv::aruco::RefineParameters &refineParams =
											cv::aruco::RefineParameters());

	bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints,
							std::vector<cv::Point2f> &imagePoints) override;
	/**
	 * @brief Returns the associated ChArUco board.
	 * @return Reference to the board.
	 */
	const cv::aruco::CharucoBoard &getBoard() const;

private:
	cv::Ptr<cv::aruco::CharucoDetector> detectorImpl_;
};

} // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_FIDUCIAL_DETECTORS_HPP
