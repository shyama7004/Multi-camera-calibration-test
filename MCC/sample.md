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
         */
        void computeRelativePose(
            const cv::Mat &rvec1, const cv::Mat &tvec1,
            const cv::Mat &rvec2, const cv::Mat &tvec2,
            cv::Mat &R_rel, cv::Mat &t_rel);

        /**
         * @brief Compute per-frame RMS reprojection errors. Each returned entry corresponds
         *        to the RMS error for a single calibration image.
         * @param objectPoints 3D corners per frame
         * @param imagePoints 2D corners per frame
         * @param cameraMatrix Intrinsics
         * @param distCoeffs Distortion
         * @param rvecs Per-frame rotation
         * @param tvecs Per-frame translation
         * @return A vector of RMS errors, one entry per frame.
         */
        std::vector<double> computePerFrameErrors(
            const std::vector<std::vector<cv::Point3f>> &objectPoints,
            const std::vector<std::vector<cv::Point2f>> &imagePoints,
            const cv::Mat &cameraMatrix,
            const cv::Mat &distCoeffs,
            const std::vector<cv::Mat> &rvecs,
            const std::vector<cv::Mat> &tvecs);

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
            CV_Assert(objectPoints.size() == imagePoints.size());
            CV_Assert(objectPoints.size() == rvecs.size());
            CV_Assert(objectPoints.size() == tvecs.size());

            double totalErr = 0.0;
            size_t totalPoints = 0;
            std::vector<cv::Point2f> projected;

            for (size_t i = 0; i < objectPoints.size(); ++i)
            {
                cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i],
                                  cameraMatrix, distCoeffs, projected);

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
            cv::Mat R1, R2;
            cv::Rodrigues(rvec1, R1);
            cv::Rodrigues(rvec2, R2);

            cv::Mat R1_inv = R1.t();
            R_rel = R2 * R1_inv;
            t_rel = tvec2 - R_rel * tvec1;
        }

        std::vector<double> computePerFrameErrors(
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

            std::vector<double> perFrameErrors;
            perFrameErrors.reserve(objectPoints.size());

            std::vector<cv::Point2f> projected;

            // Compute an RMS error for each calibration image.
            for (size_t i = 0; i < objectPoints.size(); ++i)
            {
                cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i],
                                  cameraMatrix, distCoeffs, projected);

                double err = norm(imagePoints[i], projected, cv::NORM_L2);
                double rms = std::sqrt((err * err) / objectPoints[i].size());
                perFrameErrors.push_back(rms);
            }
            return perFrameErrors;
        }

    } // namespace ccalib
} // namespace cv



```
</details>

<details>
<summary> multicam_calibrator.hpp </summary>


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


#ifndef OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP
#define OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP

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
         * @brief Holds the final intrinsics and extrinsics for a single camera.
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
         * @brief High-level manager for calibrating multiple cameras.
         *
         * This class can load configuration for one or several cameras (via a single YAML/JSON file),
         * detect calibration patterns in images, compute intrinsic and extrinsic calibration,
         * and generate both text and image-based reports.
         */
        class CV_EXPORTS MultiCameraCalibrator
        {
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
             * The file should have a “cameras” map with each camera’s config, for example:
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
             * @brief Generate a minimal text-based or image-based analysis report.
             * @param outputDir Directory where the report image will be saved.
             */
            void generateReport(const std::string &outputDir) const;

            /**
             * @brief Validate calibration across cameras by cross-checking reprojected corners, etc.
             */
            void validateCalibration() const;

            /**
             * @brief Enable or disable visualization.
             */
            void setDisplay(bool enable) { display_ = enable; }

        private:
            // Internal data structure for each camera.
            struct CameraData
            {
                int cameraId = -1;
                std::string patternType; // e.g., "chessboard", "charuco", "circles", "aruco", etc.
                cv::Size patternSize;
                float squareSize = 0.f;   // For chessboard, circle grid, etc.
                float markerSize = 0.f;   // For charuco, aruco
                std::string dictionary;   // e.g., "DICT_6X6_250"
                int calibFlags = 0;       // Calibration flags (default 0)

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
            // Relative poses: for each camera (except the reference), store (R, t) from the reference camera.
            std::map<int, std::pair<cv::Mat, cv::Mat>> realPoses_;

            bool display_ = true;

            // Loads a configuration from a file (for a single camera).
            void loadConfig(const std::string &path, CameraData &data);

            // Loads configuration data from a FileNode (used by both single and multi-camera configs).
            void loadConfigFromNode(const cv::FileNode &node, CameraData &data);

            // Detect calibration patterns in all images.
            void detectAllPatterns();

            // Calibrate intrinsics for each camera.
            void calibrateIntrinsics();

            // Compute relative poses (extrinsics) of cameras with respect to the reference camera.
            void computeRelativePoses();

            // Refine calibration using bundle adjustment, you can implement it according to your wish.
            void refineWithBundleAdjustment();
        };

    } // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_MULTICAM_CALIBRATOR_HPP


```

</details>

<details>
    <summary> multicam_calibrator.cpp </summary>

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
#include "opencv2/ccalib/multicam_calibrator.hpp"
#include "opencv2/ccalib/fiducial_detectors.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"
#include "opencv2/ccalib/visualizer.hpp"
#include <iostream>
#include <sys/stat.h>
#include <map>

namespace
{
    // Global constant for the default dictionary.
    const cv::aruco::PredefinedDictionaryType DEFAULT_DICT = cv::aruco::DICT_4X4_50;

    // Helper function to compute the relative rotation and translation from the reference camera.
    void computeRelativePose(const cv::Mat &rRef, const cv::Mat &tRef,
                             const cv::Mat &rCam, const cv::Mat &tCam,
                             cv::Mat &R_rel, cv::Mat &t_rel)
    {
        cv::Mat R_ref, R_cam;
        cv::Rodrigues(rRef, R_ref);
        cv::Rodrigues(rCam, R_cam);
        R_rel = R_cam * R_ref.t();
        t_rel = tCam - R_rel * tRef;
    }

    // Helper: Map a dictionary name string to a predefined ArUco dictionary.
    cv::Ptr<cv::aruco::Dictionary> getArucoDictionary(const std::string &dictName)
    {
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
        if (it != dictMap.end())
        {
            return cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(it->second));
        }
        // Use the default dictionary if not found.
        return cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(DEFAULT_DICT));
    }

    // Helper: Load the dictionary based on a provided dictionary string.
    cv::Ptr<cv::aruco::Dictionary> loadDictionary(const std::string &dictStr)
    {
        return getArucoDictionary(dictStr);
    }
}

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
            if (data.cameraId < 0)
            {
                std::cerr << "Invalid camera_id in " << configPath << std::endl;
                return;
            }
            if (referenceCameraId_ < 0)
            {
                referenceCameraId_ = data.cameraId;
            }
            cameraDatas_[data.cameraId] = data;
        }

        void MultiCameraCalibrator::loadMultiCameraConfig(const std::string &configPath)
        {
            cv::FileStorage fs(configPath, cv::FileStorage::READ);
            if (!fs.isOpened())
            {
                std::cerr << "Cannot open multi-camera config: " << configPath << std::endl;
                return;
            }
            cv::FileNode camerasNode = fs["cameras"];
            if (camerasNode.type() != cv::FileNode::MAP)
            {
                std::cerr << "Invalid format: 'cameras' node is not a map." << std::endl;
                return;
            }
            for (cv::FileNodeIterator it = camerasNode.begin(); it != camerasNode.end(); ++it)
            {
                CameraData data;
                loadConfigFromNode(*it, data);
                if (data.cameraId < 0)
                {
                    std::cerr << "Invalid camera_id in one of the config nodes." << std::endl;
                    continue;
                }
                if (referenceCameraId_ < 0)
                {
                    referenceCameraId_ = data.cameraId;
                }
                cameraDatas_[data.cameraId] = data;
            }
            fs.release();
        }

        void MultiCameraCalibrator::loadConfigFromNode(const cv::FileNode &node, CameraData &data)
        {
            data.cameraId = (int)node["camera_id"];
            data.patternType = (std::string)node["pattern_type"];

            // Validate and load pattern_size (must be a sequence of 2 numbers)
            cv::FileNode ps = node["pattern_size"];
            if (ps.type() == cv::FileNode::SEQ && ps.size() == 2)
            {
                data.patternSize.width = (int)ps[0];
                data.patternSize.height = (int)ps[1];
            }
            else
            {
                std::cerr << "Invalid or missing pattern_size for camera " << data.cameraId << std::endl;
                data.cameraId = -1;
                return;
            }

            data.squareSize = (float)node["square_size"];
            data.markerSize = (float)node["marker_size"];
            data.dictionary = (std::string)node["dictionary"];
            // Optional calibration flags; default is 0.
            data.calibFlags = (int)node["calib_flags"];

            cv::FileNode ipaths = node["image_paths"];
            if (ipaths.type() == cv::FileNode::SEQ)
            {
                for (cv::FileNodeIterator it = ipaths.begin(); it != ipaths.end(); ++it)
                {
                    data.imagePaths.push_back((std::string)*it);
                }
            }
        }

        void MultiCameraCalibrator::loadConfig(const std::string &path, CameraData &data)
        {
            cv::FileStorage fs(path, cv::FileStorage::READ);
            if (!fs.isOpened())
            {
                std::cerr << "Cannot open config: " << path << std::endl;
                return;
            }
            // Assuming the file itself is a map for one camera.
            loadConfigFromNode(fs.root(), data);
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
                    detector = cv::makePtr<ChessboardDetector>(cData.patternSize, cData.squareSize);
                }
                else if (cData.patternType == "circles")
                {
                    detector = cv::makePtr<CircleGridDetector>(cData.patternSize, cData.squareSize, false);
                }
                else if (cData.patternType == "acircles")
                {
                    detector = cv::makePtr<CircleGridDetector>(cData.patternSize, cData.squareSize, true);
                }
                else if (cData.patternType == "aruco")
                {
                    cv::Ptr<cv::aruco::Dictionary> dictPtr = loadDictionary(cData.dictionary);
                    detector = cv::makePtr<ArucoDetector>(dictPtr, cData.markerSize, cv::aruco::DetectorParameters());
                }
                else if (cData.patternType == "charuco")
                {
                    cv::Ptr<cv::aruco::Dictionary> dictPtr = loadDictionary(cData.dictionary);
                    cv::aruco::CharucoBoard board(cData.patternSize, cData.squareSize, cData.markerSize, *dictPtr);
                    detector = cv::makePtr<CharucoDetector>(board);
                }
                else
                {
                    std::cerr << "Unknown pattern type: " << cData.patternType << std::endl;
                    continue;
                }

                for (const auto &imgPath : cData.imagePaths)
                {
                    cv::Mat img = cv::imread(imgPath);
                    if (img.empty())
                    {
                        std::cerr << "Cannot read: " << imgPath << std::endl;
                        continue;
                    }
                    std::vector<cv::Point3f> objPts;
                    std::vector<cv::Point2f> imgPts;
                    bool ok = detector->detect(img, objPts, imgPts);
                    if (ok && !imgPts.empty())
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
                    }
                    else
                    {
                        std::cerr << "No detections in " << imgPath << std::endl;
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

        void MultiCameraCalibrator::computeRelativePoses()
        {
            if (cameraCalib_.find(referenceCameraId_) == cameraCalib_.end())
            {
                std::cerr << "Reference camera not found in calibration results\n";
                return;
            }
            const auto &refRes = cameraCalib_.at(referenceCameraId_);
            if (refRes.rvecs.empty())
                return;

            cv::Mat rRef = refRes.rvecs[0], tRef = refRes.tvecs[0];
            realPoses_[referenceCameraId_] = {cv::Mat::eye(3, 3, CV_64F),
                                              cv::Mat::zeros(3, 1, CV_64F)};

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
            // Not implemented; placeholder for future refinement.
            std::cout << "[INFO] refineWithBundleAdjustment() not implemented.\n";
        }

        void MultiCameraCalibrator::calibrate(bool computeRelative)
        {
            detectAllPatterns();
            calibrateIntrinsics();

            if (computeRelative)
            {
                computeRelativePoses();
            }

            // ------------------------------------------------------------------
            // NEW CODE for producing the error plots for each camera
            // ------------------------------------------------------------------
            CalibrationVisualizer viz;

            for (const auto &kv : cameraCalib_)
            {
                int cid = kv.first;
                const auto &calRes = kv.second;

                // Safety check
                if (calRes.rvecs.empty() || calRes.tvecs.empty())
                    continue;

                // Gather the per-frame errors
                auto &cData = cameraDatas_.at(cid); // the stored object/image points
                std::vector<double> perFrameErrors = computePerFrameErrors(
                    cData.allObjPoints, cData.allImgPoints,
                    calRes.cameraMatrix, calRes.distCoeffs,
                    calRes.rvecs, calRes.tvecs);

                // 1) Make a histogram image
                cv::Mat histImg = viz.plotErrorHistogram(perFrameErrors, 15, 800, 600);

                if (display_)
                {
                    std::string wname = " Histogram for Error Distribution of Camera" + std::to_string(cid);
                    cv::namedWindow(wname, cv::WINDOW_NORMAL);
                    cv::resizeWindow(wname, 800, 600);
                    cv::imshow(wname, histImg);
                }

                // 2) Per-frame error line plot
                cv::Mat linePlot = viz.plotReprojErrorsLine(perFrameErrors, 800, 600);

                if (display_)
                {
                    std::string wname2 = "Reprojection Errors for camera" + std::to_string(cid);
                    cv::namedWindow(wname2, cv::WINDOW_NORMAL);
                    cv::resizeWindow(wname2, 800, 600);
                    cv::imshow(wname2, linePlot);
                }
            }

            // 3) Show 3D extrinsics
            if (cameraCalib_.size() > 1)
            {
                viz.plotExtrinsics3D(cameraCalib_, referenceCameraId_, realPoses_);
            }

            if (display_)
            {
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
        }

        void MultiCameraCalibrator::generateReport(const std::string &outputDir) const
        {
            // Print text-based summary to the console.
            std::cout << "\n===== Multi-Camera Calibration Report =====\n";
            std::cout << "Reference camera: " << referenceCameraId_ << "\n";
            for (const auto &kv : cameraCalib_)
            {
                std::cout << "Camera " << kv.first << " -> ReprojErr=" << kv.second.reprojectionError << "\n";
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
            for (const auto &kv : cameraCalib_)
            {
                int cid = kv.first;
                // Check if we have the corresponding camera data.
                auto it = cameraDatas_.find(cid);
                if (it == cameraDatas_.end())
                    continue;
                const CameraData &cData = it->second;

                // Compute per-frame errors (this function is assumed available).
                std::vector<double> perFrameErrors = computePerFrameErrors(
                    cData.allObjPoints, cData.allImgPoints,
                    kv.second.cameraMatrix, kv.second.distCoeffs,
                    kv.second.rvecs, kv.second.tvecs);

                // Generate the histogram image.
                cv::Mat histImg = viz.plotErrorHistogram(perFrameErrors, 15, 800, 600);
                std::string histFilename = plotFolder + "/cam" + std::to_string(cid) + "_histogram.png";
                cv::imwrite(histFilename, histImg);

                // Generate the per-frame error line plot.
                cv::Mat linePlot = viz.plotReprojErrorsLine(perFrameErrors, 800, 600);
                std::string lineFilename = plotFolder + "/cam" + std::to_string(cid) + "_reprojErrors.png";
                cv::imwrite(lineFilename, linePlot);
            }
        }

        void MultiCameraCalibrator::saveResults(const std::string &outputPath) const
        {
            cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
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
                fs << ("cam_" + std::to_string(cid) + "_R") << kv.second.first;
                fs << ("cam_" + std::to_string(cid) + "_t") << kv.second.second;
            }
            fs << "}";
            fs.release();
            std::cout << "Saved multi-camera calibration to " << outputPath << std::endl;
        }

        void MultiCameraCalibrator::validateCalibration() const
        {
            std::cout << "[ValidateCalibration] Placeholder: implement advanced checks.\n";
        }

    } // namespace ccalib
} // namespace cv


```
</details>

<details>
    <summary> visualizer.hpp </summary>

```cpp
#ifndef OPENCV_CCALIB_VISUALIZER_HPP
#define OPENCV_CCALIB_VISUALIZER_HPP

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/multicam_calibrator.hpp>

namespace cv
{
    namespace ccalib
    {

        class CV_EXPORTS CalibrationVisualizer
        {
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
            void plotExtrinsics3D(const std::map<int, CameraCalibrationResult> &calibResults,
                                  int referenceCameraId,
                                  const std::map<int, std::pair<cv::Mat, cv::Mat>> &realPoses,
                                  const std::string &windowName = "MultiCamera Extrinsics") const;

            /**
             * @brief Produce an image showing how the lens warps a synthetic grid.
             */
            cv::Mat drawDistortionGrid(const CameraCalibrationResult &res,
                                       int gridSize,
                                       cv::Size imageSize) const;

            /**
             * @brief A histogram of errors with axis labels for “Reprojection Error (px)” vs. “Frequency”.
             */
            cv::Mat plotErrorHistogram(const std::vector<double> &errors,
                                       int histSize = 30,
                                       int histWidth = 400,
                                       int histHeight = 300) const;

            /**
             * @brief A simple line plot for per-frame errors (x-axis = frame index, y-axis = error).
             */
            cv::Mat plotReprojErrorsLine(const std::vector<double> &errors,
                                         int width = 400,
                                         int height = 300) const;
        };

    } // namespace ccalib
} // namespace cv

#endif // OPENCV_CCALIB_VISUALIZER_HPP


```
</details>

<details>
    <summary> visualizer.cpp </summary>

```cpp
#include "precomp.hpp"
#include <opencv2/ccalib/visualizer.hpp>
#include <sstream>

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

                // Green = detected corner
                cv::circle(overlay, dPt, 3, cv::Scalar(0, 255, 0), -1);
                // Red line from reprojected -> detected
                cv::line(overlay, rPt, dPt, cv::Scalar(0, 0, 255), 2);
                // Blue = reprojected corner
                cv::circle(overlay, rPt, 3, cv::Scalar(226, 211, 5), -1);
            }
            return overlay;
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
            double range = maxVal - minVal;
            if (range < 1e-12)
                range = 1e-12;

            // Count how many errors fall into each bin.
            std::vector<int> bins(histSize, 0);
            double binWidth = range / static_cast<double>(histSize);
            for (double e : errors)
            {
                int idx = static_cast<int>((e - minVal) / binWidth);
                if (idx >= histSize)
                    idx = histSize - 1;
                bins[idx]++;
            }

            // Find max bin count for y-axis range.
            int maxCount = 0;
            for (int c : bins)
                maxCount = std::max(maxCount, c);

            // Define margins, change it according to your need.
            int leftMargin = 50;
            int bottomMargin = 50;
            int topMargin = 30;
            int rightMargin = 30;

            int plotW = histWidth - leftMargin - rightMargin;
            int plotH = histHeight - topMargin - bottomMargin;

            // Create a white canvas.
            cv::Mat hist(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));

            // Scale factor for bin heights.
            double scale = static_cast<double>(plotH) / maxCount;

            // The width (in pixels) of each bar.
            int binPixWidth = cvRound(static_cast<double>(plotW) / histSize);

            // Draw each bar in blue (BGR: 255,0,0).
            for (int i = 0; i < histSize; i++)
            {
                int count = bins[i];
                int barHeight = cvRound(count * scale);

                int x1 = leftMargin + i * binPixWidth;
                int y1 = topMargin + plotH - barHeight;
                int x2 = x1 + binPixWidth - 1;
                int y2 = topMargin + plotH;

                cv::rectangle(hist,
                              cv::Point(x1, y1),
                              cv::Point(x2, y2),
                              cv::Scalar(255, 0, 0),
                              cv::FILLED);
            }

            // Draw the Y-axis (frequency) and X-axis (error).
            cv::line(hist,
                     cv::Point(leftMargin, topMargin),
                     cv::Point(leftMargin, topMargin + plotH),
                     cv::Scalar(0, 0, 0), 2);
            cv::line(hist,
                     cv::Point(leftMargin, topMargin + plotH),
                     cv::Point(leftMargin + plotW, topMargin + plotH),
                     cv::Scalar(0, 0, 0), 2);

            // X-axis tick labels: from minVal to maxVal
            int xTicks = 5;
            for (int i = 0; i <= xTicks; i++)
            {
                double tickValue = minVal + i * (range / xTicks);
                int x = leftMargin + cvRound((tickValue - minVal) / range * plotW);

                // Tick mark
                cv::line(hist,
                         cv::Point(x, topMargin + plotH),
                         cv::Point(x, topMargin + plotH + 5),
                         cv::Scalar(0, 0, 0), 1);

                // Label
                cv::putText(hist,
                            cv::format("%.2f", tickValue),
                            cv::Point(x - 15, topMargin + plotH + 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
            }

            // Y-axis tick labels: integer frequencies from 0 up to maxCount
            int yTicks = 5; // number of major ticks
            for (int i = 0; i <= yTicks; i++)
            {
                int tickVal = cvRound((maxCount / (double)yTicks) * i);
                int y = topMargin + plotH - cvRound(tickVal * scale);

                // Tick mark
                cv::line(hist,
                         cv::Point(leftMargin - 5, y),
                         cv::Point(leftMargin, y),
                         cv::Scalar(0, 0, 0), 1);

                // Label (shifted left to avoid overlapping bars)
                cv::putText(hist,
                            std::to_string(tickVal),
                            cv::Point(leftMargin - 45, y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
            }

            // Axis labels

            cv::putText(hist, "Reprojection Error (X - axis)",
                        cv::Point(leftMargin + plotW / 4, topMargin + plotH + 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            cv::putText(hist, "Frequency (Y - axis)",
                        cv::Point(leftMargin - 45, topMargin - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            return hist;
        }

        cv::Mat CalibrationVisualizer::plotReprojErrorsLine(const std::vector<double> &errors,
                                                            int width,
                                                            int height) const
        {
            // Create a white canvas.
            cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
            if (errors.empty())
                return plot;

            double minVal, maxVal;
            cv::minMaxLoc(errors, &minVal, &maxVal);
            double range = maxVal - minVal;
            if (range < 1e-12)
                range = 1e-12;

            // Define margins.
            int leftMargin = 50;
            int bottomMargin = 40;
            int topMargin = 20;
            int rightMargin = 20;
            int plotW = width - leftMargin - rightMargin;
            int plotH = height - topMargin - bottomMargin;

            // Draw X and Y axes.
            cv::line(plot, cv::Point(leftMargin, topMargin),
                     cv::Point(leftMargin, topMargin + plotH), cv::Scalar(0, 0, 0), 2);
            cv::line(plot, cv::Point(leftMargin, topMargin + plotH),
                     cv::Point(leftMargin + plotW, topMargin + plotH), cv::Scalar(0, 0, 0), 2);

            int n = static_cast<int>(errors.size());

            // Add X-axis tick labels (frame indices).
            int xTicks = std::min(n, 5);
            for (int i = 0; i < xTicks; i++)
            {
                int x = leftMargin + static_cast<int>((double)i / (xTicks - 1) * plotW);
                cv::line(plot, cv::Point(x, topMargin + plotH),
                         cv::Point(x, topMargin + plotH + 5), cv::Scalar(0, 0, 0), 1);
                cv::putText(plot, std::to_string(i),
                            cv::Point(x - 10, topMargin + plotH + 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            }

            // Add Y-axis tick labels (error values).
            int yTicks = 5;
            for (int i = 0; i <= yTicks; i++)
            {
                double value = minVal + i * (range / yTicks);
                int y = topMargin + plotH - static_cast<int>(((value - minVal) / range) * plotH);
                cv::line(plot, cv::Point(leftMargin - 5, y),
                         cv::Point(leftMargin, y), cv::Scalar(0, 0, 0), 1);
                cv::putText(plot, cv::format("%.2f", value),
                            cv::Point(5, y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            }

            // Plot the error line using blue color.
            for (int i = 0; i < n - 1; i++)
            {
                double normY1 = (errors[i] - minVal) / range;
                double normY2 = (errors[i + 1] - minVal) / range;
                int x1 = leftMargin + static_cast<int>((double)i / (n - 1) * plotW);
                int x2 = leftMargin + static_cast<int>((double)(i + 1) / (n - 1) * plotW);
                int y1 = topMargin + plotH - static_cast<int>(normY1 * plotH);
                int y2 = topMargin + plotH - static_cast<int>(normY2 * plotH);
                cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2),
                         cv::Scalar(255, 0, 0), 2);
                cv::circle(plot, cv::Point(x1, y1), 2, cv::Scalar(255, 0, 0), -1);
            }
            if (n > 1)
            {
                double normY = (errors.back() - minVal) / range;
                int xLast = leftMargin + plotW;
                int yLast = topMargin + plotH - static_cast<int>(normY * plotH);
                cv::circle(plot, cv::Point(xLast, yLast), 2, cv::Scalar(255, 0, 0), -1);
            }

            // Axis labels.
            cv::putText(plot, "Frame Index (X - axis)",
                        cv::Point(leftMargin + plotW / 3, topMargin + plotH + bottomMargin - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            cv::putText(plot, "Error (Y - axis)",
                        cv::Point(5, topMargin - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            return plot;
        }

        cv::Mat CalibrationVisualizer::drawDistortionGrid(const CameraCalibrationResult &res,
                                                          int gridSize,
                                                          cv::Size imageSize) const
        {
            cv::Mat output(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

            // Generate ideal grid points
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

            std::vector<cv::Point2f> distorted;
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

            cv::projectPoints(idealPoints, rvec, tvec, res.cameraMatrix, res.distCoeffs, distorted);

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

            // Generate a unique filename using the current time to ensure each image is saved uniquely.
            std::time_t now = std::time(nullptr);
            std::ostringstream filename;
            filename << "distortion_grid_" << now << ".png";

            // Save the image to file ensuring the full image is visible
            cv::imwrite(filename.str(), output);

            return output;
        }

        void CalibrationVisualizer::plotExtrinsics3D(
            const std::map<int, CameraCalibrationResult> &calibResults,
            int /*referenceCameraId*/,
            const std::map<int, std::pair<cv::Mat, cv::Mat>> &realPoses,
            const std::string &windowName) const
        {
#ifdef HAVE_OPENCV_VIZ
            // Create a Viz3d window
            cv::viz::Viz3d vizWin(windowName);

            // Set background to black
            vizWin.setBackgroundColor(cv::viz::Color::black());

            // Show a small coordinate system at the origin (scale = 0.1)
            vizWin.showWidget("GlobalCoord", cv::viz::WCoordinateSystem(0.1));

            // Register a keyboard callback for zoom in ('+') and zoom out ('-')
            vizWin.registerKeyboardCallback([](const cv::viz::KeyboardEvent &event, void *cookie)
                                            {
                cv::viz::Viz3d* win = static_cast<cv::viz::Viz3d*>(cookie);
                // Only act on key press (not release)
                if (event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN)
                {
                    // Get the current camera pose
                    cv::Affine3d camPose = win->getViewerPose();
                    cv::Vec3d pos = camPose.translation();
                    // Zoom in: move the camera 10% closer to the origin
                    if (event.code == '+')
                    {
                        pos *= 0.9;
                        camPose.translation() = pos;
                        win->setViewerPose(camPose);
                    }
                    // Zoom out: move the camera 10% farther from the origin
                    else if (event.code == '-')
                    {
                        pos *= 1.1;
                        camPose.translation() = pos;
                        win->setViewerPose(camPose);
                    }
                } }, &vizWin);

            // Loop through each camera's extrinsics
            for (const auto &kv : realPoses)
            {
                int camId = kv.first;
                const cv::Mat &R = kv.second.first;  // 3x3
                const cv::Mat &t = kv.second.second; // 3x1

                // Convert R,t into an "inverted" pose for displaying the camera
                cv::Mat R_inv = R.t();
                cv::Mat t_inv = -R_inv * t;

                cv::Matx33d R33(R_inv);
                cv::Vec3d tVec(t_inv);

                cv::Affine3d pose(R33, tVec);

                // A small coordinate system for each camera
                std::string widgetName = "Camera_" + std::to_string(camId);
                cv::viz::WCameraPosition camCoord(0.05); // smaller axis length
                vizWin.showWidget(widgetName + "_coord", camCoord, pose);

                // If intrinsics exist, also draw a frustum (in yellow)
                auto it = calibResults.find(camId);
                if (it != calibResults.end())
                {
                    const CameraCalibrationResult &cres = it->second;
                    CV_Assert(cres.cameraMatrix.type() == CV_64F &&
                              cres.cameraMatrix.size() == cv::Size(3, 3));
                    cv::Matx33d K(cres.cameraMatrix);
                    // Slightly smaller scale for the frustum; note: the color is explicitly set to yellow.
                    cv::viz::WCameraPosition frustum(K, 0.05, cv::viz::Color::yellow());
                    vizWin.showWidget(widgetName + "_frustum", frustum, pose);
                }

                // Label each camera with a smaller font size
                // (the last parameter is the font size in world units)
                cv::viz::WText3D text3d("cam " + std::to_string(camId),
                                        cv::Point3d(0, 0, 0.03), // small offset
                                        0.02,
                                        true,
                                        cv::viz::Color::white());
                vizWin.showWidget(widgetName + "_label", text3d, pose);
            }

            // Start interactive 3D visualization
            vizWin.spin();

#else
            (void)calibResults;
            (void)realPoses;
            (void)windowName;
            std::cout << "[Warning] OpenCV built without Viz module; 3D visualization not available.\n";
#endif
        }

    } // namespace ccalib
} // namespace cv


```
</details>
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
