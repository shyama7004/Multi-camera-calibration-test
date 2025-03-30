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
│   ├── multicam_calibrator.hpp
│   └── visualizer.hpp
├── src/
│   ├── fiducial_detectors.cpp
│   ├── calibration_utils.cpp
│   ├── multicam_calibrator.cpp
│   └── visualizer.cpp
└── samples/
│   ├── camera1.yaml
│   ├── camera2.yaml
    └── multicam_calib.cpp


```

### Components added :

<details> <summary>
    fiducial_detectors.hpp </summary>

```cpp
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
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
        /**
         * @brief Base class for fiducial detectors.
         */
        class CV_EXPORTS FiducialDetector
        {
        public:
            virtual ~FiducialDetector() {}
            /**
             * @brief Detect fiducial points in the given image.
             * @param image Input image.
             * @param objectPoints Output 3D object points.
             * @param imagePoints Output 2D image points.
             * @return true if detection was successful.
             */
            virtual bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) = 0;
        };

        /**
         * @brief Detector for chessboard patterns.
         */
        class CV_EXPORTS ChessboardDetector : public FiducialDetector
        {
        public:
            /**
             * @brief Constructs a ChessboardDetector.
             * @param patternSize Number of inner corners per chessboard row and column.
             * @param squareSize Size of a square in user-defined units.
             * @param detectionFlags Flags for corner detection.
             * @param subPixCriteria Termination criteria for corner refinement.
             */
            ChessboardDetector(cv::Size patternSize, float squareSize,
                               int detectionFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK,
                               cv::TermCriteria subPixCriteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) override;

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
        class CV_EXPORTS ArucoDetector : public FiducialDetector
        {
        public:
            /**
             * @brief Constructs an ArucoDetector.
             * @param dictionary Aruco dictionary.
             * @param markerLength Length of the marker's side.
             * @param detectorParams Detector parameters.
             */
            ArucoDetector(const cv::Ptr<cv::aruco::Dictionary> &dictionary, float markerLength, const cv::aruco::DetectorParameters &detectorParams = cv::aruco::DetectorParameters());

            bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) override;

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
        class CV_EXPORTS CircleGridDetector : public FiducialDetector
        {
        public:
            /**
             * @brief Constructs a CircleGridDetector.
             * @param patternSize Number of circles per row and column.
             * @param circleSize Diameter of a circle.
             * @param asymmetric Flag indicating whether the grid is asymmetric.
             */
            CircleGridDetector(cv::Size patternSize, float circleSize, bool asymmetric);

            bool detect(cv::InputArray image, std::vector<cv::Point3f> &objectPoints, std::vector<cv::Point2f> &imagePoints) override;

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
        class CV_EXPORTS CharucoDetector : public FiducialDetector
        {
        public:
            /**
             * @brief Constructs a CharucoDetector.
             * @param board The ChArUco board.
             * @param charucoParams Parameters specific to ChArUco detection.
             * @param detectorParams General Aruco detector parameters.
             * @param refineParams Parameters for refining detections.
             */
            CharucoDetector(const cv::aruco::CharucoBoard &board,
                            const cv::aruco::CharucoParameters &charucoParams = cv::aruco::CharucoParameters(),
                            const cv::aruco::DetectorParameters &detectorParams = cv::aruco::DetectorParameters(),
                            const cv::aruco::RefineParameters &refineParams = cv::aruco::RefineParameters());

            bool detect(cv::InputArray image,
                        std::vector<cv::Point3f> &objectPoints,
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


```

</details>
<details> <summary> fiducial_detectors.cpp
</summary>

```cpp
/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
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

#include "precomp.hpp"
#include <opencv2/ccalib/fiducial_detectors.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>

namespace cv
{
    namespace ccalib
    {
        // this is a helper function to convert an image to grayscale.
        namespace
        {
            cv::Mat toGray(const cv::Mat &image)
            {
                if (image.channels() > 1)
                {
                    cv::Mat gray;
                    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
                    return gray;
                }
                return image;
            }
        }

        // Build a 3D chessboard object points vector based on board size and square size.
        static std::vector<cv::Point3f> buildChessboard3D(cv::Size sz, float sqSize)
        {
            std::vector<cv::Point3f> pts;
            pts.reserve(sz.width * sz.height);
            // Loop over rows and columns to create object points.
            for (int rows = 0; rows < sz.height; ++rows)
            {
                for (int cols = 0; cols < sz.width; ++cols)
                {
                    pts.emplace_back(cols * sqSize, rows * sqSize, 0.f);
                }
            }
            return pts;
        }

        // Constructor for ChessboardDetector: initializes parameters and precomputed object points.
        ChessboardDetector::ChessboardDetector(cv::Size patternSize, float squareSize, int detectionFlags, cv::TermCriteria subPixCriteria)
            : patternSize_(patternSize), squareSize_(squareSize),
              detectionFlags_(detectionFlags), subPixCriteria_(subPixCriteria),
              precomputedObjectPoints_(buildChessboard3D(patternSize, squareSize))
        {
        }
        // Set the detection flags for the chessboard detector.
        void ChessboardDetector::setDetectionFlags(int flags)
        {
            detectionFlags_ = flags;
        }

        // Set the criteria for sub-pixel corner refinement.
        void ChessboardDetector::setSubPixCriteria(cv::TermCriteria criteria)
        {
            subPixCriteria_ = criteria;
        }

        // Detect chessboard corners in the input image.
        bool ChessboardDetector::detect(cv::InputArray inImage,
                                        std::vector<cv::Point3f> &objectPoints,
                                        std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();
            if (image.empty())
            {
                CV_LOG_WARNING(NULL, "Input image is empty");
                return false;
            }

            cv::Mat grayImage = toGray(image);

            bool found = findChessboardCorners(grayImage, patternSize_, imagePoints, detectionFlags_);

            if (!found)
            {
                CV_LOG_WARNING(NULL, "Chessboard corners not found");
                return false;
            }

            // Refine corner positions using sub-pixel accuracy.
            cornerSubPix(grayImage, imagePoints, cv::Size(11, 11), cv::Size(-1, -1), subPixCriteria_);
            objectPoints = precomputedObjectPoints_;
            return true;
        }

        // Build a 3D circle grid object points vector based on grid size, circle distance, and asymmetry flag.
        static std::vector<cv::Point3f> buildCircle3D(cv::Size sz, float d, bool asym)
        {
            std::vector<cv::Point3f> pts;
            pts.reserve(sz.width * sz.height);
            // Loop over rows and columns, you can adjust offset for asymmetric grid.
            for (int rows = 0; rows < sz.height; ++rows)
            {
                for (int cols = 0; cols < sz.width; ++cols)
                {
                    float offset = (asym && (rows % 2 == 1)) ? d * 0.5f : 0.f;
                    pts.emplace_back(cols * d + offset, rows * d, 0.f);
                }
            }
            return pts;
        }

        // Constructor for CircleGridDetector: initializes parameters and precomputed object points.
        CircleGridDetector::CircleGridDetector(cv::Size patternSize, float circleSize, bool asymmetric)
            : patternSize_(patternSize), circleSize_(circleSize), asymmetric_(asymmetric),
              precomputedObjectPoints_(buildCircle3D(patternSize, circleSize, asymmetric))
        {
        }

        // Detect circle grid corners in the input image.
        bool CircleGridDetector::detect(cv::InputArray inImage,
                                        std::vector<cv::Point3f> &objectPoints,
                                        std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();
            if (image.empty())
            {
                CV_LOG_WARNING(NULL, "Input image is empty");
                return false;
            }

            // Use helper function for grayscale conversion.
            cv::Mat grayImage = toGray(image);

            // Set appropriate flags based on whether grid is asymmetric.
            int flags = asymmetric_ ? (cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING)
                                    : cv::CALIB_CB_SYMMETRIC_GRID;

            bool found = findCirclesGrid(grayImage, patternSize_, imagePoints, flags);
            if (!found)
            {
                CV_LOG_WARNING(NULL, "Circle grid corners not found");
                return false;
            }

            objectPoints = precomputedObjectPoints_;
            return true;
        }

        // Constructor for CharucoDetector: initializes the underlying ArUco Charuco detector.
        CharucoDetector::CharucoDetector(const cv::aruco::CharucoBoard &board,
                                         const cv::aruco::CharucoParameters &charucoParams,
                                         const cv::aruco::DetectorParameters &detectorParams,
                                         const cv::aruco::RefineParameters &refineParams)
        {
            detectorImpl_ = cv::makePtr<cv::aruco::CharucoDetector>(board, charucoParams, detectorParams, refineParams);
        }

        // Detect Charuco board in the input image.
        bool CharucoDetector::detect(cv::InputArray inImage,
                                     std::vector<cv::Point3f> &objectPoints,
                                     std::vector<cv::Point2f> &imagePoints)
        {
            cv::Mat image = inImage.getMat();

            if (image.empty())
            {
                CV_LOG_WARNING(NULL, "Input image is empty");
                return false;
            }
            cv::Mat grayImage = toGray(image);

            std::vector<cv::Point2f> charucoCorners;
            std::vector<int> charucoIds;

            // Detect Charuco board corners and corresponding IDs.
            detectorImpl_->detectBoard(grayImage, charucoCorners, charucoIds);

            // Verify that corners and IDs are detected.
            if (charucoCorners.empty() || charucoIds.empty())
            {
                CV_LOG_WARNING(NULL, "Charuco corners or IDs not found");
                return false;
            }

            detectorImpl_->getBoard().matchImagePoints(charucoCorners, charucoIds, objectPoints, imagePoints);
            if (objectPoints.empty() || imagePoints.empty())
            {
                CV_LOG_WARNING(NULL, "No matching points found");
                return false;
            }

            return true;
        }

        // Get the associated Charuco board.
        const cv::aruco::CharucoBoard &CharucoDetector::getBoard() const
        {
            return detectorImpl_->getBoard();
        }

        // Constructor for ArucoDetector: initializes dictionary, marker length, and detector parameters.
        ArucoDetector::ArucoDetector(const cv::Ptr<cv::aruco::Dictionary> &dictionary, float markerLength,
                                     const cv::aruco::DetectorParameters &detectorParams)
            : dictionary_(dictionary), markerLength_(markerLength), detectorParams_(detectorParams) {}

        // Build a 3D marker object points vector based on marker length.
        static std::vector<cv::Point3f> buildMarker3D(float markerLength)
        {
            return {
                {0.f, 0.f, 0.f},
                {markerLength, 0.f, 0.f},
                {markerLength, markerLength, 0.f},
                {0.f, markerLength, 0.f}};
        }

        // Detect ArUco markers in the input image.
        bool ArucoDetector::detect(cv::InputArray inImage,
                                   std::vector<cv::Point3f> &objectPoints,
                                   std::vector<cv::Point2f> &imagePoints)
        {
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

            for (size_t i = 0; i < markerIds.size(); i++)
            {
                objectPoints.insert(objectPoints.end(), markerObjPts.begin(), markerObjPts.end());
                imagePoints.insert(imagePoints.end(), markerCorners[i].begin(), markerCorners[i].end());
            }

            return true;
        }

        // Get the dictionary used for marker detection.
        const cv::Ptr<cv::aruco::Dictionary> &ArucoDetector::getDictionary() const
        {
            return dictionary_;
        }

        // Get the marker length parameter.
        float ArucoDetector::getMarkerLength() const
        {
            return markerLength_;
        }

    } // namespace ccalib
} // namespace cv


```
</details>

<details>
    <summary> calibration_utils.hpp </summary>

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
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
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

#ifndef OPENCV_CCALIB_CALIBRATION_UTILS_HPP
#define OPENCV_CCALIB_CALIBRATION_UTILS_HPP

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
</details>

<details>
    <summary> calibration_utils.cpp </summary>

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
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
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

#include "precomp.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"

namespace cv
{
    namespace ccalib
    {

        /**
         * @brief Computes the average reprojection error for a set of views.
         *
         * @param objectPoints 3D object points in world coordinates.
         * @param imagePoints 2D image points corresponding to objectPoints.
         * @param cameraMatrix Intrinsic parameters of the camera.
         * @param distCoeffs Distortion coefficients.
         * @param rvecs Rotation vectors for each view.
         * @param tvecs Translation vectors for each view.
         * @return Average reprojection error.
         */
        double computeReprojectionError(
            const std::vector<std::vector<cv::Point3f>> &objectPoints,
            const std::vector<std::vector<cv::Point2f>> &imagePoints,
            const cv::Mat &cameraMatrix,
            const cv::Mat &distCoeffs,
            const std::vector<cv::Mat> &rvecs,
            const std::vector<cv::Mat> &tvecs)
        {
            CV_Assert(objectPoints.size() == imagePoints.size());
            CV_Assert(objectPoints.size() == rvecs.size());
            CV_Assert(objectPoints.size() == tvecs.size());

            double totalErr = 0.0;
            size_t totalPoints = 0;
            std::vector<cv::Point2f> projected; // I declared it outside the loop to avoid reallocating it in each iteration

            for (size_t i = 0; i < objectPoints.size(); ++i)
            {
                cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projected);

                double err = norm(imagePoints[i], projected, cv::NORM_L2);
                totalErr += err * err;
                totalPoints += objectPoints[i].size();
            }

            return std::sqrt(totalErr / totalPoints);
        }

        /**
         * @brief Computes the relative rotation and translation between two camera poses.
         *
         * @param rvec1 Rotation vector of camera 1.
         * @param tvec1 Translation vector of camera 1.
         * @param rvec2 Rotation vector of camera 2.
         * @param tvec2 Translation vector of camera 2.
         * @param R_rel Output: relative rotation matrix from camera 1 to camera 2.
         * @param t_rel Output: relative translation vector from camera 1 to camera 2.
         */
        void computeRelativePose(
            const cv::Mat &rvec1, const cv::Mat &tvec1,
            const cv::Mat &rvec2, const cv::Mat &tvec2,
            cv::Mat &R_rel, cv::Mat &t_rel)
        {
            cv::Mat R1, R2;
            cv::Rodrigues(rvec1, R1);
            cv::Rodrigues(rvec2, R2);

            const cv::Mat R1_inv = R1.t(); // inverse of R1
            R_rel = R2 * R1_inv;
            t_rel = tvec2 - R_rel * tvec1;
        }

    } // namespace ccalib
} // namespace cv


```
</details>

5. multicam_calibrator.hpp


```cpp
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
        struct CV_EXPORTS CameraCalibrationResult
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
        class CV_EXPORTS MultiCameraCalibrator
        {
        public:
            MultiCameraCalibrator();
            ~MultiCameraCalibrator();

            /**
             * @brief Load a single camera dataset from a config file (YAML/JSON).
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
            std::map<int, std::pair<cv::Mat, cv::Mat>> realPoses_;

            bool display_ = true;

        private:
            void loadConfig(const std::string &path, CameraData &data);
            void detectAllPatterns();
            void calibrateIntrinsics();
            void computeRelativePoses();
            void refineWithBundleAdjustment();
        };

    } // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP


```

6. multicam_calibrator.cpp

```cpp
/**
 * @file multicam_calibrator.cpp
 * @brief Implementation of MultiCameraCalibrator for multi-fiducial calibration.
 */

#include "opencv2/ccalib/multicam_calibrator.hpp"
#include "opencv2/ccalib/fiducial_detectors.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"

// Include our new visualizer
#include "opencv2/ccalib/visualizer.hpp"

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

            // 5) (Optional) visualize results (2D error maps, 3D extrinsics, distortion grid).
            //    This is new, courtesy of the Visualizer module.
            {
                CalibrationVisualizer viz;

                // (A) Visualize 2D reprojection errors for the reference camera on the first frame
                auto itRef = cameraCalib_.find(referenceCameraId_);
                if (itRef != cameraCalib_.end())
                {
                    const CameraCalibrationResult &refRes = itRef->second;

                    // Check if we have at least 1 frame for the ref camera
                    const auto &dataRef = cameraDatas_.at(referenceCameraId_);
                    if (!dataRef.allObjPoints.empty())
                    {
                        // We'll reproject the corners from the first frame
                        const std::vector<cv::Point3f> &objPts = dataRef.allObjPoints[0];
                        const std::vector<cv::Point2f> &detectedPts = dataRef.allImgPoints[0];

                        // Reproject
                        std::vector<cv::Point2f> reprojectedPts;
                        cv::projectPoints(objPts,
                                          refRes.rvecs[0], refRes.tvecs[0],
                                          refRes.cameraMatrix, refRes.distCoeffs,
                                          reprojectedPts);

                        // Load the actual image for that frame if desired
                        cv::Mat refImg;
                        if (!dataRef.imagePaths.empty())
                        {
                            refImg = cv::imread(dataRef.imagePaths[0]);
                        }
                        if (refImg.empty())
                        {
                            // fallback: white image if the file can't be loaded
                            refImg = cv::Mat(dataRef.imageSize, CV_8UC3, cv::Scalar(255, 255, 255));
                        }

                        // Draw reprojection error map
                        cv::Mat errorOverlay = viz.drawReprojErrorMap(refImg, detectedPts, reprojectedPts);
                        cv::imshow("Reference Camera Reproj Error", errorOverlay);
                        cv::waitKey(500); // Show briefly
                    }

                    // (B) Distortion grid for the ref camera
                    cv::Mat distGrid = viz.drawDistortionGrid(refRes, 10, refRes.imageSize);
                    cv::imshow("Ref Camera Distortion Grid", distGrid);
                    cv::waitKey(500);
                }

                // (C) If we have multiple cameras, show 3D extrinsics (requires opencv_viz)
                if (cameraCalib_.size() > 1)
                {
                    viz.plotExtrinsics3D(cameraCalib_, referenceCameraId_, realPoses_);
                }

                // (Optional) You could do more advanced visual checks here
                // e.g. histograms, multi-camera cross-views, etc.

                // Finally, close any windows
                cv::destroyAllWindows();
            }
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
            for (const auto &kv : realPoses_)
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
            std::cout << "\n===== Multi-Camera Calibration Report =====\n";
            std::cout << "Reference camera: " << referenceCameraId_ << "\n";
            for (const auto &kv : cameraCalib_)
            {
                std::cout << "Camera " << kv.first << " -> ReprojErr=" << kv.second.reprojectionError << "\n";
            }
            std::cout << "===========================================\n\n";

            cv::Mat white(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(white, "Calibration Summary", cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            cv::imwrite(outputDir + "/calib_report.png", white);
        }

        void MultiCameraCalibrator::validateCalibration() const
        {
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

            data.squareSize = (float)fs["square_size"];
            data.markerSize = (float)fs["marker_size"];
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
            for (auto &kv : cameraDatas_)
            {
                auto &cData = kv.second;

                cv::Ptr<FiducialDetector> detector;
                if (cData.patternType == "chessboard")
                {
                    detector = makePtr<ChessboardDetector>(cData.patternSize, cData.squareSize);
                }
                else if (cData.patternType == "circles")
                {
                    detector = makePtr<CircleGridDetector>(cData.patternSize, cData.squareSize, false);
                }
                else if (cData.patternType == "acircles")
                {
                    detector = makePtr<CircleGridDetector>(cData.patternSize, cData.squareSize, true);
                }
                else if (cData.patternType == "aruco")
                {
                    cv::Ptr<cv::aruco::Dictionary> dictPtr = cv::makePtr<cv::aruco::Dictionary>(
                        (!cData.dictionary.empty()) ? cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
                                                    : cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
                    detector = makePtr<ArucoDetector>(dictPtr, cData.markerSize, cv::aruco::DetectorParameters());
                }
                else if (cData.patternType == "charuco")
                {
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
                            cv::Mat displayImg = img.clone();
                            for (size_t i = 0; i < imgPts.size(); i++)
                            {
                                cv::circle(displayImg, imgPts[i], 4, cv::Scalar(0, 0, 255), -1);
                            }
                            cv::imshow("Detections", displayImg);
                            int key = cv::waitKey(300);
                            if (key == 27)
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
                    0);

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
            if (cameraCalib_.find(referenceCameraId_) == cameraCalib_.end())
            {
                std::cerr << "Ref camera not found in calibration results\n";
                return;
            }
            const auto &refRes = cameraCalib_.at(referenceCameraId_);
            if (refRes.rvecs.empty())
                return;

            cv::Mat rRef = refRes.rvecs[0], tRef = refRes.tvecs[0];
            realPoses_[referenceCameraId_] = {cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)};

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
                realPoses_[cid] = {R_rel, t_rel};
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
7. visualizer.hpp

```cpp
#ifndef OPENCV_CCALIB_VISUALIZER_HPP
#define OPENCV_CCALIB_VISUALIZER_HPP

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/multicam_calibrator.hpp>

/**
 * @file visualizer.hpp
 * @brief Visualization tools for camera calibration results.
 */

namespace cv
{
    namespace ccalib
    {

        /**
         * @class CalibrationVisualizer
         * @brief Provides methods to visualize calibration outputs in 2D or 3D.
         */
        class CV_EXPORTS CalibrationVisualizer
        {
        public:
            CalibrationVisualizer();

            /**
             * @brief Draw reprojection error vectors on top of an image.
             * @param image              The original image.
             * @param detectedCorners    The 2D detected corners in the image.
             * @param reprojectedCorners The reprojected 2D points for the same corners.
             * @return                   A copy of the input image annotated with lines showing errors.
             */
            cv::Mat drawReprojErrorMap(const cv::Mat &image,
                                       const std::vector<cv::Point2f> &detectedCorners,
                                       const std::vector<cv::Point2f> &reprojectedCorners) const;

            /**
             * @brief Visualize multi-camera extrinsics in 3D (requires opencv_viz).
             * @param calibResults       Map of cameraId -> camera calibration result
             * @param referenceCameraId  The ID of the reference camera
             * @param relPoses           Map of cameraId -> (R, t) relative to reference
             * @param windowName         Viz3d window name
             */
            void plotExtrinsics3D(const std::map<int, CameraCalibrationResult> &calibResults,
                                  int referenceCameraId,
                                  const std::map<int, std::pair<cv::Mat, cv::Mat>> &relPoses,
                                  const std::string &windowName = "MultiCamera Extrinsics") const;

            /**
             * @brief Produce an image showing how the lens warps a synthetic grid.
             * @param res        The calibration result (intrinsics, distortion).
             * @param gridSize   # lines in each dimension (e.g., 10).
             * @param imageSize  Output resolution.
             * @return           An image with a distorted grid drawn.
             */
            cv::Mat drawDistortionGrid(const CameraCalibrationResult &res,
                                       int gridSize,
                                       cv::Size imageSize) const;

            /**
             * @brief Compute and display a simple histogram of per-corner or per-frame errors.
             * @param errors      A list of numeric error values.
             * @param histSize    Number of bins.
             * @param histWidth   Width of the histogram image.
             * @param histHeight  Height of the histogram image.
             * @return            A BGR (or grayscale) image containing the histogram.
             */
            cv::Mat plotErrorHistogram(const std::vector<double> &errors,
                                       int histSize = 30,
                                       int histWidth = 400,
                                       int histHeight = 300) const;
        };

    } // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_VISUALIZER_HPP

```
8. visualizer.cpp
```cpp
#include <opencv2/ccalib/visualizer.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// If Viz is available, we'll use it
#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif

namespace cv
{
    namespace ccalib
    {

        CalibrationVisualizer::CalibrationVisualizer() {}

        cv::Mat CalibrationVisualizer::drawReprojErrorMap(const cv::Mat &image,
                                                          const std::vector<cv::Point2f> &detectedCorners,
                                                          const std::vector<cv::Point2f> &reprojectedCorners) const
        {
            CV_Assert(!image.empty());
            CV_Assert(detectedCorners.size() == reprojectedCorners.size());

            cv::Mat overlay = image.clone();
            for (size_t i = 0; i < detectedCorners.size(); i++)
            {
                cv::Point2f dPt = detectedCorners[i];
                cv::Point2f rPt = reprojectedCorners[i];

                // Draw the detected corner
                cv::circle(overlay, dPt, 3, cv::Scalar(0, 255, 0), -1);

                // Draw a line from reprojected corner -> detected corner
                cv::line(overlay, rPt, dPt, cv::Scalar(0, 0, 255), 2);

                // Draw the reprojected corner as well
                cv::circle(overlay, rPt, 3, cv::Scalar(255, 0, 0), -1);
            }
            return overlay;
        }

        void CalibrationVisualizer::plotExtrinsics3D(const std::map<int, CameraCalibrationResult> &calibResults,
                                                     int /*referenceCameraId*/, // not strictly needed in this method
                                                     const std::map<int, std::pair<cv::Mat, cv::Mat>> &relPoses,
                                                     const std::string &windowName) const
        {
#ifdef HAVE_OPENCV_VIZ
            cv::viz::Viz3d vizWin(windowName);
            vizWin.showWidget("Coord", cv::viz::WCoordinateSystem(0.2)); // global origin

            for (const auto &kv : relPoses)
            {
                int camId = kv.first;
                cv::Mat R = kv.second.first.clone();  // 3x3
                cv::Mat t = kv.second.second.clone(); // 3x1

                // Inverse so we can place the camera coordinate system in the scene
                cv::Mat R_inv = R.t();
                cv::Mat t_inv = -R_inv * t;

                // Convert these to an Affine3d
                cv::Matx33d R33(R_inv.at<double>(0, 0), R_inv.at<double>(0, 1), R_inv.at<double>(0, 2),
                                R_inv.at<double>(1, 0), R_inv.at<double>(1, 1), R_inv.at<double>(1, 2),
                                R_inv.at<double>(2, 0), R_inv.at<double>(2, 1), R_inv.at<double>(2, 2));
                cv::Vec3d tVec(t_inv.at<double>(0, 0), t_inv.at<double>(1, 0), t_inv.at<double>(2, 0));

                cv::Affine3d pose(R33, tVec);

                // Draw a small coordinate system at the camera's pose
                std::string widgetName = "Camera_" + std::to_string(camId);
                cv::viz::WCameraPosition camCoord(0.1); // axis length
                vizWin.showWidget(widgetName + "_coord", camCoord, pose);

                // If we have intrinsics, let's also draw a frustum
                auto it = calibResults.find(camId);
                if (it != calibResults.end())
                {
                    const CameraCalibrationResult &cres = it->second;
                    // Convert cameraMatrix to Matx33d
                    CV_Assert(cres.cameraMatrix.type() == CV_64F && cres.cameraMatrix.size() == cv::Size(3, 3));
                    cv::Matx33d K(cres.cameraMatrix.at<double>(0, 0), cres.cameraMatrix.at<double>(0, 1), cres.cameraMatrix.at<double>(0, 2),
                                  cres.cameraMatrix.at<double>(1, 0), cres.cameraMatrix.at<double>(1, 1), cres.cameraMatrix.at<double>(1, 2),
                                  cres.cameraMatrix.at<double>(2, 0), cres.cameraMatrix.at<double>(2, 1), cres.cameraMatrix.at<double>(2, 2));

                    // The constructor WCameraPosition(const Matx33d& K, double scale, const Color& color=Color::white());
                    cv::viz::WCameraPosition frustum(K, 0.1, cv::viz::Color::yellow());
                    vizWin.showWidget(widgetName + "_frustum", frustum, pose);
                }
            }

            vizWin.spin();
#else
            (void)calibResults; // to suppress unused warnings
            (void)relPoses;
            (void)windowName;
            std::cout << "[Warning] OpenCV was built without Viz module; 3D visualization is not available.\n";
#endif
        }

        cv::Mat CalibrationVisualizer::drawDistortionGrid(const CameraCalibrationResult &res,
                                                          int gridSize,
                                                          cv::Size imageSize) const
        {
            cv::Mat output(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

            // Generate ideal grid points as 3D points (z=0)
            std::vector<cv::Point3f> idealPoints;
            idealPoints.reserve((gridSize + 1) * (gridSize + 1));
            for (int i = 0; i <= gridSize; i++)
            {
                float y = i * (float)imageSize.height / gridSize;
                for (int j = 0; j <= gridSize; j++)
                {
                    float x = j * (float)imageSize.width / gridSize;
                    idealPoints.push_back(cv::Point3f(x, y, 0.f));
                }
            }

            // Distort them using the camera model
            std::vector<cv::Point2f> distorted;
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

            cv::projectPoints(idealPoints, rvec, tvec,
                              res.cameraMatrix, res.distCoeffs,
                              distorted);

            // Draw horizontal lines
            for (int i = 0; i <= gridSize; i++)
            {
                int rowIdx = i * (gridSize + 1);
                for (int j = 0; j < gridSize; j++)
                {
                    cv::Point2f p1 = distorted[rowIdx + j];
                    cv::Point2f p2 = distorted[rowIdx + j + 1];
                    cv::line(output, p1, p2, cv::Scalar(0, 0, 0), 1);
                }
            }
            // Draw vertical lines
            for (int j = 0; j <= gridSize; j++)
            {
                for (int i = 0; i < gridSize; i++)
                {
                    int idx1 = i * (gridSize + 1) + j;
                    int idx2 = (i + 1) * (gridSize + 1) + j;
                    cv::Point2f p1 = distorted[idx1];
                    cv::Point2f p2 = distorted[idx2];
                    cv::line(output, p1, p2, cv::Scalar(0, 0, 0), 1);
                }
            }

            return output;
        }

        cv::Mat CalibrationVisualizer::plotErrorHistogram(const std::vector<double> &errors,
                                                          int histSize,
                                                          int histWidth,
                                                          int histHeight) const
        {
            if (errors.empty())
            {
                return cv::Mat(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));
            }

            double minVal, maxVal;
            cv::minMaxLoc(errors, &minVal, &maxVal);

            // ensure non-zero range
            double range = (maxVal - minVal);
            if (range < 1e-12)
                range = 1e-12;

            std::vector<int> bins(histSize, 0);
            double binWidth = range / (double)histSize;

            for (double e : errors)
            {
                int idx = (int)((e - minVal) / binWidth);
                if (idx >= histSize)
                    idx = histSize - 1;
                bins[idx]++;
            }

            int maxCount = 0;
            for (int c : bins)
            {
                if (c > maxCount)
                    maxCount = c;
            }

            cv::Mat hist(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));
            double scale = (double)histHeight / (double)maxCount;
            int binW = histWidth / histSize;

            for (int i = 0; i < histSize; i++)
            {
                int count = bins[i];
                int h = (int)(count * scale);
                cv::rectangle(hist,
                              cv::Point(i * binW, histHeight - h),
                              cv::Point((i + 1) * binW, histHeight),
                              cv::Scalar(0, 0, 255), cv::FILLED);
            }

            return hist;
        }

    } // namespace ccalib
} // namespace cv

```
9. samples\multicam_calib.cpp

```cpp
// samples/multicam_calibration.cpp

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/ccalib/fiducial_detectors.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"
#include "opencv2/ccalib/multicam_calibrator.hpp"
#include "opencv2/ccalib/visualizer.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <string>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::ccalib;

static void help()
{
    cout << "\nThis is an updated sample for multi-camera calibration using the new MultiCameraCalibrator and CalibrationVisualizer." << endl;
    cout << "Usage:" << endl;
    cout << "    multicam_calibration <camera1.yaml> <camera2.yaml> [more.yaml...]" << endl;
    cout << "Each YAML file should contain the camera configuration (see sample YAML below)." << endl;
    cout << "\nSample YAML (camera1.yaml):" << endl;
    cout << "  camera_id: 1" << endl;
    cout << "  pattern_type: chessboard" << endl;
    cout << "  pattern_size: [9, 6]" << endl;
    cout << "  square_size: 0.025" << endl;
    cout << "  image_paths:" << endl;
    cout << "    - \"images/camera1/img1.jpg\"" << endl;
    cout << "    - \"images/camera1/img2.jpg\"" << endl;
    cout << "    - \"images/camera1/img3.jpg\"" << endl;
    cout << endl;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        help();
        return 1;
    }

    // Create our multi-camera calibrator instance
    MultiCameraCalibrator calibrator;

    // Add each camera configuration from the command-line YAML files
    for (int i = 1; i < argc; i++)
    {
        string configFile = argv[i];
        cout << "Loading configuration: " << configFile << endl;
        calibrator.addCamera(configFile);
    }

    // Optionally set display mode (to show detection windows, etc.)
    calibrator.setDisplay(true);

    // Run the calibration process: detection, individual calibration, and relative pose computation.
    calibrator.calibrate(true);

    // Save the calibration results into a YAML file.
    calibrator.saveResults("multi_calib_result.yaml");

    // Generate a simple report that includes (for demonstration) a 2D error map and writes a basic summary image.
    calibrator.generateReport("results_output");

    // --- Additional Visualizations using the new CalibrationVisualizer ---
    // Note: The class is named CalibrationVisualizer, not Visualizer.
    CalibrationVisualizer viz;

    // Example: Plot a simple histogram of dummy errors (in a real application, extract your actual error values)
    vector<double> dummyErrors = {0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.65};
    Mat histImg = viz.plotErrorHistogram(dummyErrors, 10);
    imwrite("results_output/error_histogram.png", histImg);
    imshow("Error Histogram", histImg);
    waitKey(500);

    // Example: 3D extrinsics visualization.
    // For demonstration, we build dummy calibration results and extrinsics.
    map<int, CameraCalibrationResult> dummyCalib;
    CameraCalibrationResult dummy;
    dummy.cameraMatrix = (Mat_<double>(3, 3) << 800, 0, 320,
                          0, 800, 240,
                          0, 0, 1);
    dummy.distCoeffs = (Mat_<double>(5, 1) << 0.1, -0.05, 0, 0, 0);
    dummy.imageSize = Size(640, 480);
    dummy.rvecs.push_back(Mat::zeros(3, 1, CV_64F));
    dummy.tvecs.push_back(Mat::zeros(3, 1, CV_64F));
    dummyCalib[1] = dummy;
    dummyCalib[2] = dummy;

    map<int, pair<Mat, Mat>> dummyExtrinsics;
    dummyExtrinsics[1] = {Mat::eye(3, 3, CV_64F), (Mat_<double>(3, 1) << 0, 0, 0)};
    dummyExtrinsics[2] = {Mat::eye(3, 3, CV_64F), (Mat_<double>(3, 1) << 0.2, 0, 0)};

    // This will open a Viz window (if OpenCV was built with opencv_viz)
    viz.plotExtrinsics3D(dummyCalib, 1, dummyExtrinsics, "Camera Extrinsics");

    return 0;
}

```
10. samples\camera1.yaml

```yaml
%YAML:1.0

camera_id: 1
pattern_type: chessboard
pattern_size: [8, 6]
square_size: 0.028
image_paths:
  - "/Users/sankarsanbisoyi/Desktop/Dataset/shyamas_dataset/chessboard/Extrinsics/set_0/anshu/1.jpeg"
  - "/Users/sankarsanbisoyi/Desktop/Dataset/shyamas_dataset/chessboard/Extrinsics/set_0/anshu/2.jpeg"


```
11. samples\camera2.yaml

```yaml
%YAML:1.0

camera_id: 1
pattern_type: chessboard
pattern_size: [8, 6]
square_size: 0.028
image_paths:
  - "/Users/sankarsanbisoyi/Desktop/Dataset/shyamas_dataset/chessboard/Extrinsics/set_0/f14/1.jpeg"
  - "/Users/sankarsanbisoyi/Desktop/Dataset/shyamas_dataset/chessboard/Extrinsics/set_0/f14/2.jpeg"


```
