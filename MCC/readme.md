### Progress uptil now :

#### Project description : 

We are looking for a student to curate best of class calibration data, collect calibration data with various OpenCV Fiducials, and graphically produce calibration board and camera models data (script). Simultaneously, begin to write comprehensive test scripts of all the existing calibration functions. While doing this, if necessary, improve the calibration documentation. Derive from this expected accuracy of fiducial types for various camera types.

#### Expected Outcomes:
- Curate camera calibration data from public datasets.
- Collect calibration data for various fiducials and camera types.
- Graphically create camera calibration data with ready to go scripts
- Write test functions for the OpenCV Calibration pipeline
- New/improved documentation on how to calibrate cameras as needed.
- Statistical analysis of the performance (accuracy and variance) of OpenCV fiducials, algorithms and camera types.
- A YouTube video showing describing and demonstrating the OpenCV Calibration testss.

Directory dtructure for my c++ code implementaion :


After imlementaion directory struture :

```bash
ccalib/
├── CMakeLists.txt
├── include/opencv2/ccalib/
│   ├── fiducial_detectors.hpp
│   ├── calibration_utils.hpp
│   └── multicam_calibrator.hpp
├── src/
│   ├── fiducial_detectors.cpp
│   ├── calibration_utils.cpp
│   └── multicam_calibrator.cpp
└── samples/
    └── multicam_calibration.cpp

```

### Components added :

1. fiducial_detectors.hpp

```cpp
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CCALIB_FIDUCIAL_DETECTORS_HPP
#define OPENCV_CCALIB_FIDUCIAL_DETECTORS_HPP

#include "opencv2/core.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

namespace cv
{
    namespace ccalib
    {
        class FiducialDetector
        {
        public:
            virtual ~FiducialDetector() {}
            virtual bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) = 0;
        };

        class ChessboardDetector : public FiducialDetector
        {
        public:
            ChessboardDetector(cv::Size patternSize, float squareSize);
            bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints);

        private:
            cv::Size patternSize_;
            float squareSize_;
        };

        class ArucoDetector : public FiducialDetector
        {
        public:
            ArucoDetector(const cv::Ptr<cv::aruco::Dictionary> &dictionary, float markerLength, const cv::aruco::DetectorParameters &detectorParams = cv::aruco::DetectorParameters());

            bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) override;

            const cv::Ptr<cv::aruco::Dictionary> &getDictionary() const;
            float getMarkerLength() const;

        private:
            cv::Ptr<cv::aruco::Dictionary> dictionary_;
            float markerLength_;
            cv::aruco::DetectorParameters detectorParams_;
        };

        class CircleGridDetector : public FiducialDetector
        {
        public:
            CircleGridDetector(cv::Size patternSize, float circleSize, bool asymmetric);
            bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) override;

        private:
            cv::Size patternSize_;
            float circleSize_;
            bool asymmetric_;
        };

        class CharucoDetector : public FiducialDetector
        {
        public:
            CharucoDetector(const cv::aruco::CharucoBoard &board,
                            const cv::aruco::CharucoParameters &charucoParams = cv::aruco::CharucoParameters(),
                            const cv::aruco::DetectorParameters &detectorParams = cv::aruco::DetectorParameters(),
                            const cv::aruco::RefineParameters &refineParams = cv::aruco::RefineParameters());

            bool detect(cv::InputArray image,
                        std::vector<cv::Point3f> &objectPoints,
                        std::vector<cv::Point2f> &imagePoints) override;
            const cv::aruco::CharucoBoard &getBoard() const;

        private:
            cv::Ptr<cv::aruco::CharucoDetector> detectorImpl_;
        };

    }
};     // namespace cv::ccalib
#endif // OPENCV_CCALIB_FIDUCIAL_DETECTORS_HPP
```


2. fiducial_detectors.cpp

```py

/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/ccalib/fiducial_detectors.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

namespace cv
{
    namespace ccalib
    {

        /*--------------------------------------------------
           ChessboardDetector
        --------------------------------------------------*/

        /**
         * @brief Constructor for ChessboardDetector
         * @param patternSize Number of interior corners, e.g. 8x6
         * @param squareSize Physical size of each square, e.g. 0.025
         */
        ChessboardDetector::ChessboardDetector(cv::Size patternSize, float squareSize)
            : patternSize_(patternSize),
              squareSize_(squareSize)
        {
            // empty
        }

        // Helper: build the 3D points for a standard chessboard
        static std::vector<cv::Point3f> buildChessboard3D(cv::Size sz, float sqSize)
        {
            std::vector<cv::Point3f> pts;
            pts.reserve(sz.width * sz.height);
            for (int r = 0; r < sz.height; ++r)
            {
                for (int c = 0; c < sz.width; ++c)
                {
                    pts.push_back(cv::Point3f(c * sqSize, r * sqSize, 0.f));
                }
            }
            return pts;
        }

        bool ChessboardDetector::detect(cv::InputArray inImage,
                                        std::vector<cv::Point3f> &objectPoints,
                                        std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();

            // Use standard OpenCV chessboard detection
            bool found = findChessboardCorners(image, patternSize_, imagePoints,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
            if (!found)
            {
                return false;
            }

            // Optionally refine corners
            if (image.channels() > 1)
            {
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            }
            cornerSubPix(image, imagePoints, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // Build the 3D coordinates for each corner
            objectPoints = buildChessboard3D(patternSize_, squareSize_);
            return true;
        }

        /*--------------------------------------------------
                       CircleGridDetector
        --------------------------------------------------*/

        /**
         * @brief Constructor for CircleGridDetector
         * @param patternSize total circles in X and Y
         * @param circleSize distance between circle centers
         * @param asymmetric if true, CALIB_CB_ASYMMETRIC_GRID is used
         */
        CircleGridDetector::CircleGridDetector(cv::Size patternSize, float circleSize, bool asymmetric)
            : patternSize_(patternSize),
              circleSize_(circleSize),
              asymmetric_(asymmetric)
        {
            // empty
        }

        // Helper to build 3D coords for circle grids
        static std::vector<cv::Point3f> buildCircle3D(cv::Size sz, float d, bool asym)
        {
            std::vector<cv::Point3f> pts;
            pts.reserve(sz.width * sz.height);
            for (int r = 0; r < sz.height; ++r)
            {
                for (int c = 0; c < sz.width; ++c)
                {
                    float offset = 0.f;
                    if (asym && (r % 2 == 1))
                    {
                        offset = d * 0.5f; // shift half a circle in alternate rows
                    }
                    pts.push_back(cv::Point3f(c * d + offset, r * d, 0.f));
                }
            }
            return pts;
        }

        bool CircleGridDetector::detect(cv::InputArray inImage,
                                        std::vector<cv::Point3f> &objectPoints,
                                        std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();

            int flags = 0;
            if (asymmetric_)
            {
                flags = cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING;
            }
            else
            {
                flags = cv::CALIB_CB_SYMMETRIC_GRID;
            }

            bool found = findCirclesGrid(image, patternSize_, imagePoints, flags);
            if (!found)
            {
                return false;
            }

            objectPoints = buildCircle3D(patternSize_, circleSize_, asymmetric_);
            return true;
        }

        /*--------------------------------------------------
                          CharucoDetector
        --------------------------------------------------*/

        /**
         * @brief CharucoDetector constructor that creates an official CharucoDetector instance.
         * @param board           Pre-configured Charuco board.
         * @param charucoParams   Parameters for Charuco detection (optional).
         * @param detectorParams  Parameters for marker detection (optional).
         * @param refineParams    Parameters for refining detection (optional).
         */
        CharucoDetector::CharucoDetector(const cv::aruco::CharucoBoard &board,
                                         const cv::aruco::CharucoParameters &charucoParams,
                                         const cv::aruco::DetectorParameters &detectorParams,
                                         const cv::aruco::RefineParameters &refineParams)
        {
            // Create the official CharucoDetector using the provided board and parameters
            detectorImpl_ = cv::makePtr<cv::aruco::CharucoDetector>(board, charucoParams, detectorParams, refineParams);
        }

        bool CharucoDetector::detect(cv::InputArray inImage,
                                     std::vector<cv::Point3f> &objectPoints,
                                     std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();

            // Prepare output containers for the official detector.
            cv::Mat charucoCorners, charucoIds;
            // Note: We pass noArray() for markerCorners and markerIds so that the detector
            // internally performs marker detection.
            detectorImpl_->detectBoard(image, charucoCorners, charucoIds);

            if (charucoCorners.empty() || charucoIds.empty())
            {
                return false;
            }

            // Use the CharucoBoard's matchImagePoints to generate corresponding 3D and 2D points.
            detectorImpl_->getBoard().matchImagePoints(charucoCorners, charucoIds, objectPoints, imagePoints);
            if (objectPoints.empty() || imagePoints.empty())
            {
                return false;
            }

            return true;
        }

        const cv::aruco::CharucoBoard &CharucoDetector::getBoard() const
        {
            return detectorImpl_->getBoard();
        }

        /*--------------------------------------------------
                        ArucoDetector
        --------------------------------------------------*/

        ArucoDetector::ArucoDetector(const cv::Ptr<cv::aruco::Dictionary> &dictionary, float markerLength,
                                     const cv::aruco::DetectorParameters &detectorParams)
            : dictionary_(dictionary),
              markerLength_(markerLength),
              detectorParams_(detectorParams)
        {
            // empty constructor body
        }

        bool ArucoDetector::detect(cv::InputArray inImage,
                                   std::vector<cv::Point3f> &objectPoints,
                                   std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

            // Create a local ArucoDetector instance.
            // Note: The constructor for cv::aruco::ArucoDetector expects a dictionary by reference.
            cv::aruco::ArucoDetector detector(*dictionary_, detectorParams_);
            detector.detectMarkers(image, markerCorners, markerIds, rejectedCandidates);

            if (markerIds.empty())
            {
                return false;
            }

            // For each detected marker, generate corresponding object points.
            // Assume that the marker is a square lying in the z=0 plane.
            objectPoints.clear();
            imagePoints.clear();
            for (size_t i = 0; i < markerIds.size(); i++)
            {
                // 3D object points for one marker (order: top-left, top-right, bottom-right, bottom-left)
                std::vector<cv::Point3f> markerObjPts = {
                    cv::Point3f(0, 0, 0),
                    cv::Point3f(markerLength_, 0, 0),
                    cv::Point3f(markerLength_, markerLength_, 0),
                    cv::Point3f(0, markerLength_, 0)};

                // Append the 3D points and corresponding 2D image points.
                objectPoints.insert(objectPoints.end(), markerObjPts.begin(), markerObjPts.end());
                imagePoints.insert(imagePoints.end(), markerCorners[i].begin(), markerCorners[i].end());
            }

            return true;
        }

        const cv::Ptr<cv::aruco::Dictionary> &ArucoDetector::getDictionary() const
        {
            return dictionary_;
        }

        float ArucoDetector::getMarkerLength() const
        {
            return markerLength_;
        }

    } // namespace ccalib
} // namespace cv

```

3. calibration_utils.hpp

```py

#ifndef OPENCV_CCALIB_CALIBRATION_UTILS_HPP
#define OPENCV_CCALIB_CALIBRATION_UTILS_HPP

/**
 * @file calibration_utils.hpp
 * @brief Utility functions for computing reprojection error and relative poses.
 */

#include <opencv2/core.hpp>
#include <vector>

namespace cv
{
    namespace ccalib
    {

        /**
         * @brief Compute average reprojection error across multiple frames.
         * @param objectPoints 3D points for each frame.
         * @param imagePoints 2D corners for each frame.
         * @param cameraMatrix Intrinsic matrix (3x3).
         * @param distCoeffs Distortion coefficients.
         * @param rvecs Per-frame rotation vectors.
         * @param tvecs Per-frame translation vectors.
         * @return The RMS reprojection error in pixels.
         */
        double computeReprojectionError(
            const std::vector<std::vector<cv::Point3f>> &objectPoints,
            const std::vector<std::vector<cv::Point2f>> &imagePoints,
            const cv::Mat &cameraMatrix,
            const cv::Mat &distCoeffs,
            const std::vector<cv::Mat> &rvecs,
            const std::vector<cv::Mat> &tvecs);

        /**
         * @brief Compute relative pose (R_rel, t_rel) from camera1 to camera2, given their extrinsics in the world frame.
         * R_rel = R2 * R1^T
         * t_rel = t2 - R_rel * t1
         */
        void computeRelativePose(
            const cv::Mat &rvec1, const cv::Mat &tvec1,
            const cv::Mat &rvec2, const cv::Mat &tvec2,
            cv::Mat &R_rel, cv::Mat &t_rel);

    } // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_CALIBRATION_UTILS_HPP
```
4. calibration_utils.cpp

```py

/**
 * @file calibration_utils.cpp
 * @brief Implementation of reprojection error and relative pose computations.
 */

#include "opencv2/ccalib/calibration_utils.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>

namespace cv
{
    namespace ccalib
    {

        double computeReprojectionError(
            const std::vector<std::vector<cv::Point3f>> &objectPoints,
            const std::vector<std::vector<cv::Point2f>> &imagePoints,
            const cv::Mat &cameraMatrix,
            const cv::Mat &distCoeffs,
            const std::vector<cv::Mat> &rvecs,
            const std::vector<cv::Mat> &tvecs)
        {
            double totalErr = 0.0;
            size_t totalPoints = 0;

            for (size_t i = 0; i < objectPoints.size(); i++)
            {
                std::vector<cv::Point2f> projected;
                cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projected);

                double err = norm(imagePoints[i], projected, cv::NORM_L2);
                totalErr += err * err;
                totalPoints += objectPoints[i].size();
            }

            return std::sqrt(totalErr / totalPoints);
        }

        void computeRelativePose(
            const cv::Mat &rvec1, const cv::Mat &tvec1,
            const cv::Mat &rvec2, const cv::Mat &tvec2,
            cv::Mat &R_rel, cv::Mat &t_rel)
        {
            // Convert rvec->R
            cv::Mat R1, R2;
            Rodrigues(rvec1, R1);
            Rodrigues(rvec2, R2);

            R_rel = R2 * R1.t();
            t_rel = tvec2 - R_rel * tvec1;
        }

    } // namespace ccalib
} // namespace cv


```

5. multicam_calib.hpp


```py

#ifndef OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP
#define OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP

/**
 * @file multicam_calibrator.hpp
 * @brief A class that loads camera configs, detects patterns, calibrates each camera, and computes relative poses.
 */

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

namespace cv
{
    namespace ccalib
    {

        /**
         * @struct CameraCalibrationResult
         * @brief Holds the final intrinsics + extrinsics for a single camera.
         */
        struct CameraCalibrationResult
        {
            cv::Mat cameraMatrix;
            cv::Mat distCoeffs;
            std::vector<cv::Mat> rvecs;
            std::vector<cv::Mat> tvecs;
            double reprojectionError = -1.0;
            cv::Size imageSize;
        };

        /**
         * @class MultiCameraCalibrator
         * @brief High-level manager for calibrating multiple cameras from separate config files.
         */
        class MultiCameraCalibrator
        {
        public:
            MultiCameraCalibrator();
            ~MultiCameraCalibrator();

            /**
             * @brief Load a single camera dataset from a config file (YAML/JSON).
             * The config must contain:
             *  - camera_id (int)
             *  - pattern_type ("chessboard", "charuco", "circles", etc.)
             *  - pattern_size [width, height]
             *  - square_size or marker_size
             *  - image_paths (array of file paths)
             *  - dictionary (for charuco, optional)
             */
            void addCamera(const std::string &configPath);

            /**
             * @brief Calibrate each camera individually and optionally compute relative extrinsics.
             * @param computeRelative If true, computeRelativePoses() is called using the first camera as reference.
             */
            void calibrate(bool computeRelative = true);

            /**
             * @brief Save intrinsics and extrinsics for all cameras to a single file.
             * @param outputPath path to the output YAML
             */
            void saveResults(const std::string &outputPath) const;

            /**
             * @brief Generate a minimal text-based or image-based analysis.
             * E.g. print errors, produce plots.
             */
            void generateReport(const std::string &outputDir) const;

            /**
             * @brief Validate calibration across cameras by cross-checking reprojected corners, etc.
             */
            void validateCalibration() const;

            void setDisplay(bool enable) { display_ = enable; }

        private:
            // internal data for each camera
            struct CameraData
            {
                int cameraId = -1;
                std::string patternType; // "chessboard", "charuco", "circles", ...
                cv::Size patternSize;
                float squareSize = 0.f; // e.g. for chessboard or circle
                float markerSize = 0.f; // e.g. for charuco
                std::string dictionary; // e.g. "DICT_6X6_250"

                std::vector<std::string> imagePaths;

                // accumulations
                std::vector<std::vector<cv::Point3f>> allObjPoints;
                std::vector<std::vector<cv::Point2f>> allImgPoints;
                cv::Size imageSize;
            };

            // camera ID -> data
            std::map<int, CameraData> cameraDatas_;
            // camera ID -> final calibration
            std::map<int, CameraCalibrationResult> cameraCalib_;

            // reference camera ID
            int referenceCameraId_ = -1;
            // relPoses_[camId] = (R, t) from reference camera to this camera
            std::map<int, std::pair<cv::Mat, cv::Mat>> relPoses_;

            bool display_ = true;

        private:
            // helpers
            void loadConfig(const std::string &path, CameraData &data);
            void detectAllPatterns();
            void calibrateIntrinsics();
            void computeRelativePoses();

            // optional BA if needed
            void refineWithBundleAdjustment();
        };

    } // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP

```

6. multicam_calib.cpp

```py

/**
 * @file multicam_calibrator.cpp
 * @brief Implementation of MultiCameraCalibrator for multi-fiducial calibration.
 */

#include "opencv2/ccalib/multicam_calib.hpp"
#include "opencv2/ccalib/fiducial_detectors.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <mutex>

namespace cv
{
    namespace ccalib
    {

        MultiCameraCalibrator::MultiCameraCalibrator() {}
        MultiCameraCalibrator::~MultiCameraCalibrator() {}

        void MultiCameraCalibrator::addCamera(const std::string &configPath)
        {
            CameraData data;
            loadConfig(configPath, data);

            if (referenceCameraId_ < 0)
            {
                // the first camera loaded becomes reference
                referenceCameraId_ = data.cameraId;
            }
            cameraDatas_[data.cameraId] = data;
        }

        void MultiCameraCalibrator::calibrate(bool computeRelative)
        {
            // 1) Detect corners/markers
            detectAllPatterns();

            // 2) Calibrate each camera
            calibrateIntrinsics();

            // 3) Optionally compute extrinsics relative to reference camera
            if (computeRelative)
            {
                computeRelativePoses();
            }

            // 4) (Optional) refine globally
            // refineWithBundleAdjustment();
        }

        void MultiCameraCalibrator::saveResults(const std::string &outputPath) const
        {
            FileStorage fs(outputPath, FileStorage::WRITE);
            if (!fs.isOpened())
            {
                std::cerr << "Cannot open file for writing: " << outputPath << std::endl;
                return;
            }
            fs << "reference_camera_id" << referenceCameraId_;

            fs << "cameras" << "[";
            for (const auto &kv : cameraCalib_)
            {
                int camId = kv.first;
                const auto &cRes = kv.second;
                fs << "{:"
                   << "camera_id" << camId
                   << "camera_matrix" << cRes.cameraMatrix
                   << "dist_coeffs" << cRes.distCoeffs
                   << "reproj_error" << cRes.reprojectionError
                   << "}";
            }
            fs << "]";
            fs << "relative_poses" << "{";
            for (const auto &kv : relPoses_)
            {
                int cid = kv.first;
                fs << "cam_" + std::to_string(cid) + "_R" << kv.second.first;
                fs << "cam_" + std::to_string(cid) + "_t" << kv.second.second;
            }
            fs << "}";

            fs.release();
            std::cout << "Saved multi-camera calibration to " << outputPath << std::endl;
        }

        void MultiCameraCalibrator::generateReport(const std::string &outputDir) const
        {
            // minimal console-based reporting
            std::cout << "\n===== Multi-Camera Calibration Report =====\n";
            std::cout << "Reference camera: " << referenceCameraId_ << "\n";
            for (const auto &kv : cameraCalib_)
            {
                std::cout << "Camera " << kv.first << " -> ReprojErr=" << kv.second.reprojectionError << "\n";
            }
            std::cout << "===========================================\n\n";

            // Example: create a trivial image to show
            cv::Mat white(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(white, "Calibration Summary", cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            cv::imwrite(outputDir + "/calib_report.png", white);
        }

        void MultiCameraCalibrator::validateCalibration() const
        {
            // For advanced cross-checking, you can reproject known corners from cameraA to cameraB's frame.
            // We'll just print a placeholder.
            std::cout << "[ValidateCalibration] Placeholder: implement advanced checks.\n";
        }

        // -----------------------------------------------------
        // Private / helper
        // -----------------------------------------------------
        void MultiCameraCalibrator::loadConfig(const std::string &path, CameraData &data)
        {
            FileStorage fs(path, FileStorage::READ);
            if (!fs.isOpened())
            {
                std::cerr << "Cannot open config: " << path << std::endl;
                return;
            }
            data.cameraId = (int)fs["camera_id"];
            data.patternType = (std::string)fs["pattern_type"];

            {
                FileNode ps = fs["pattern_size"];
                if (ps.type() == FileNode::SEQ && ps.size() == 2)
                {
                    data.patternSize.width = (int)ps[0];
                    data.patternSize.height = (int)ps[1];
                }
            }

            data.squareSize = (float)fs["square_size"]; // e.g. for chessboard or circle grid
            data.markerSize = (float)fs["marker_size"]; // e.g. for charuco
            data.dictionary = (std::string)fs["dictionary"];

            FileNode ipaths = fs["image_paths"];
            if (ipaths.type() == FileNode::SEQ)
            {
                for (FileNodeIterator it = ipaths.begin(); it != ipaths.end(); ++it)
                {
                    data.imagePaths.push_back((std::string)*it);
                }
            }

            fs.release();
        }

        void MultiCameraCalibrator::detectAllPatterns()
        {
            // We'll do single-threaded detection for simplicity.
            // For each camera, for each image, detect corners.
            for (auto &kv : cameraDatas_)
            {
                auto &cData = kv.second;

                // Build the correct fiducial detector
                cv::Ptr<FiducialDetector> detector;
                if (cData.patternType == "chessboard")
                {
                    detector = makePtr<ChessboardDetector>(cData.patternSize, cData.squareSize);
                }
                else if (cData.patternType == "circles")
                {
                    detector = makePtr<CircleGridDetector>(cData.patternSize, cData.squareSize, /*asymmetric=*/false);
                }
                else if (cData.patternType == "acircles")
                {
                    detector = makePtr<CircleGridDetector>(cData.patternSize, cData.squareSize, /*asymmetric=*/true);
                }
                else if (cData.patternType == "aruco")
                {
                    // Create the dictionary pointer using the configuration data.
                    cv::Ptr<cv::aruco::Dictionary> dictPtr = cv::makePtr<cv::aruco::Dictionary>(
                        (!cData.dictionary.empty()) ? cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
                                                    : cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
                    // Create the ArucoDetector (ensure you have defined squares_x, squares_y, etc. in your config if needed)
                    detector = makePtr<ArucoDetector>(dictPtr, cData.markerSize, cv::aruco::DetectorParameters());
                }

                else if (cData.patternType == "charuco")
                {
                    // parse dictionary name
                    cv::Ptr<cv::aruco::Dictionary> dictPtr = cv::makePtr<cv::aruco::Dictionary>(
                        (!cData.dictionary.empty()) ? cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
                                                    : cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
                    cv::aruco::CharucoBoard board(cData.patternSize, cData.squareSize, cData.markerSize, *dictPtr);
                    detector = makePtr<CharucoDetector>(board);
                }
                else
                {
                    std::cerr << "Unknown pattern type: " << cData.patternType << std::endl;
                    continue;
                }

                for (const auto &imgPath : cData.imagePaths)
                {
                    cv::Mat img = imread(imgPath);
                    if (img.empty())
                    {
                        std::cerr << "Cannot read: " << imgPath << std::endl;
                        continue;
                    }
                    std::vector<cv::Point3f> objPts;
                    std::vector<cv::Point2f> imgPts;
                    bool ok = detector->detect(img, objPts, imgPts);
                    if (ok)
                    {
                        cData.allObjPoints.push_back(objPts);
                        cData.allImgPoints.push_back(imgPts);
                        cData.imageSize = img.size();

                        if (display_)
                        {
                            // We'll draw a circle for each 2D corner in imgPts
                            cv::Mat displayImg = img.clone();
                            for (size_t i = 0; i < imgPts.size(); i++)
                            {
                                cv::circle(displayImg, imgPts[i], 4, cv::Scalar(0, 0, 255), -1);
                            }
                            // Show the result briefly
                            cv::imshow("Detections", displayImg);
                            int key = cv::waitKey(300); // ~0.3 second so user can see
                            if (key == 27)              // ESC key
                                break;
                        }
                        else if (imgPts.empty())
                        {
                            std::cerr << "No detections in " << imgPath << std::endl;
                        }
                    }
                }
            }
            if (display_)
            {
                cv::destroyWindow("Detections");
            }
        }

        void MultiCameraCalibrator::calibrateIntrinsics()
        {
            // For each camera that has detections, run calibrateCamera
            for (auto &kv : cameraDatas_)
            {
                int cid = kv.first;
                auto &cData = kv.second;

                if (cData.allObjPoints.empty())
                {
                    std::cerr << "No detections for camera " << cid << std::endl;
                    continue;
                }

                CameraCalibrationResult res;
                cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
                cv::Mat dist = cv::Mat::zeros(8, 1, CV_64F);
                std::vector<cv::Mat> rvecs, tvecs;

                double rms = cv::calibrateCamera(
                    cData.allObjPoints,
                    cData.allImgPoints,
                    cData.imageSize,
                    K, dist,
                    rvecs, tvecs,
                    0 // flags, adjust as needed
                );

                double reproj = computeReprojectionError(cData.allObjPoints, cData.allImgPoints,
                                                         K, dist, rvecs, tvecs);

                res.cameraMatrix = K.clone();
                res.distCoeffs = dist.clone();
                res.rvecs = rvecs;
                res.tvecs = tvecs;
                res.reprojectionError = reproj;
                res.imageSize = cData.imageSize;
                cameraCalib_[cid] = res;

                std::cout << "[Camera " << cid << "] RMS=" << rms
                          << ", ReprojErr=" << reproj << std::endl;
            }
        }

        void MultiCameraCalibrator::computeRelativePoses()
        {
            // pick reference's rvecs[0], tvecs[0]
            if (cameraCalib_.find(referenceCameraId_) == cameraCalib_.end())
            {
                std::cerr << "Ref camera not found in calibration results\n";
                return;
            }
            const auto &refRes = cameraCalib_.at(referenceCameraId_);
            if (refRes.rvecs.empty())
                return;

            cv::Mat rRef = refRes.rvecs[0], tRef = refRes.tvecs[0];
            relPoses_[referenceCameraId_] = {cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)};

            for (const auto &kv : cameraCalib_)
            {
                int cid = kv.first;
                if (cid == referenceCameraId_)
                    continue;
                const auto &cRes = kv.second;
                if (cRes.rvecs.empty())
                    continue;

                cv::Mat R_rel, t_rel;
                computeRelativePose(rRef, tRef, cRes.rvecs[0], cRes.tvecs[0], R_rel, t_rel);
                relPoses_[cid] = {R_rel, t_rel};
            }
        }

        void MultiCameraCalibrator::refineWithBundleAdjustment()
        {
            // not implemented
            std::cout << "[INFO] refineWithBundleAdjustment() not implemented.\n";
        }

    } // namespace ccalib
} // namespace cv

```
### Futrher improvements that is needed

1. 2D Reprojection Error Maps

- After calibrating, you should have for each image a set of reprojected corners vs. detected corners.
- You should plot these errors as vectors on the image or color-coded heatmaps to see if certain corners (e.g., edges vs. center) have higher distortion.

2. 3D Camera Extrinsic Visualization

In a multi‐camera setup, you have each camera’s rotation (R) and translation (t) relative to some reference.
- You can plot small “camera pyramids” or “frustums” in 3D space (using a 3D library like matplotlib’s Axes3D in Python or something like Viz3d in OpenCV).
- This visually confirms if cameras’ relative positions make sense.

3. Distortion Plots

- If you have k1, k2, p1, p2, …, you can illustrate how lines or grid points get warped by the lens.
- Some people like to produce a radial plot showing how far from the center rays get bent.
- Statistical Charts

4. Histograms of per-image or per-camera reprojection error.

- Box plots of errors for different fiducial types (checkerboard vs. ArUco vs. Charuco vs circles).
- Aggregated charts comparing cameras or lens models.
- Synthetic “Rendered” Scenes


5. Improve the earlier code based on opencv docs, there a lots of redundancy.
