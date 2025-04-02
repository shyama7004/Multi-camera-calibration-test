// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/ccalib/visualizer.hpp>
#include <sstream>

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif

namespace cv {
namespace ccalib {

CalibrationVisualizer::CalibrationVisualizer() {}

cv::Mat CalibrationVisualizer::drawReprojErrorMap(
    const cv::Mat &image, const std::vector<cv::Point2f> &detectedCorners,
    const std::vector<cv::Point2f> &reprojectedCorners) const {
  CV_Assert(!image.empty());
  CV_Assert(detectedCorners.size() == reprojectedCorners.size());

  cv::Mat overlay = image.clone();
  for (size_t i = 0; i < detectedCorners.size(); i++) {
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
cv::Mat
CalibrationVisualizer::plotErrorHistogram(const std::vector<double> &errors,
                                          int histSize, int histWidth,
                                          int histHeight) const {
  if (errors.empty()) {
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
  for (double e : errors) {
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
  for (int i = 0; i < histSize; i++) {
    int count = bins[i];
    int barHeight = cvRound(count * scale);

    int x1 = leftMargin + i * binPixWidth;
    int y1 = topMargin + plotH - barHeight;
    int x2 = x1 + binPixWidth - 1;
    int y2 = topMargin + plotH;

    cv::rectangle(hist, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(255, 0, 0), cv::FILLED);
  }

  // Draw the Y-axis (frequency) and X-axis (error).
  cv::line(hist, cv::Point(leftMargin, topMargin),
           cv::Point(leftMargin, topMargin + plotH), cv::Scalar(0, 0, 0), 2);
  cv::line(hist, cv::Point(leftMargin, topMargin + plotH),
           cv::Point(leftMargin + plotW, topMargin + plotH),
           cv::Scalar(0, 0, 0), 2);

  // X-axis tick labels: from minVal to maxVal
  int xTicks = 5;
  for (int i = 0; i <= xTicks; i++) {
    double tickValue = minVal + i * (range / xTicks);
    int x = leftMargin + cvRound((tickValue - minVal) / range * plotW);

    // Tick mark
    cv::line(hist, cv::Point(x, topMargin + plotH),
             cv::Point(x, topMargin + plotH + 5), cv::Scalar(0, 0, 0), 1);

    // Label
    cv::putText(hist, cv::format("%.2f", tickValue),
                cv::Point(x - 15, topMargin + plotH + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
  }

  // Y-axis tick labels: integer frequencies from 0 up to maxCount
  int yTicks = 5; // number of major ticks
  for (int i = 0; i <= yTicks; i++) {
    int tickVal = cvRound((maxCount / (double)yTicks) * i);
    int y = topMargin + plotH - cvRound(tickVal * scale);

    // Tick mark
    cv::line(hist, cv::Point(leftMargin - 5, y), cv::Point(leftMargin, y),
             cv::Scalar(0, 0, 0), 1);

    // Label (shifted left to avoid overlapping bars)
    cv::putText(hist, std::to_string(tickVal),
                cv::Point(leftMargin - 45, y + 5), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(0, 0, 0), 1);
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

cv::Mat
CalibrationVisualizer::plotReprojErrorsLine(const std::vector<double> &errors,
                                            int width, int height) const {
  // I have used a white background for the plot.
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
           cv::Point(leftMargin + plotW, topMargin + plotH),
           cv::Scalar(0, 0, 0), 2);

  int n = static_cast<int>(errors.size());

  // Add X-axis tick labels (frame indices).
  int xTicks = std::min(n, 5);
  for (int i = 0; i < xTicks; i++) {
    int x = leftMargin + static_cast<int>((double)i / (xTicks - 1) * plotW);
    cv::line(plot, cv::Point(x, topMargin + plotH),
             cv::Point(x, topMargin + plotH + 5), cv::Scalar(0, 0, 0), 1);
    cv::putText(plot, std::to_string(i),
                cv::Point(x - 10, topMargin + plotH + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
  }

  // Add Y-axis tick labels (error values).
  int yTicks = 5;
  for (int i = 0; i <= yTicks; i++) {
    double value = minVal + i * (range / yTicks);
    int y = topMargin + plotH -
            static_cast<int>(((value - minVal) / range) * plotH);
    cv::line(plot, cv::Point(leftMargin - 5, y), cv::Point(leftMargin, y),
             cv::Scalar(0, 0, 0), 1);
    cv::putText(plot, cv::format("%.2f", value), cv::Point(5, y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
  }

  // Plot the error line using blue color.
  for (int i = 0; i < n - 1; i++) {
    double normY1 = (errors[i] - minVal) / range;
    double normY2 = (errors[i + 1] - minVal) / range;
    int x1 = leftMargin + static_cast<int>((double)i / (n - 1) * plotW);
    int x2 = leftMargin + static_cast<int>((double)(i + 1) / (n - 1) * plotW);
    int y1 = topMargin + plotH - static_cast<int>(normY1 * plotH);
    int y2 = topMargin + plotH - static_cast<int>(normY2 * plotH);
    cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0),
             2);
    cv::circle(plot, cv::Point(x1, y1), 2, cv::Scalar(255, 0, 0), -1);
  }
  if (n > 1) {
    double normY = (errors.back() - minVal) / range;
    int xLast = leftMargin + plotW;
    int yLast = topMargin + plotH - static_cast<int>(normY * plotH);
    cv::circle(plot, cv::Point(xLast, yLast), 2, cv::Scalar(255, 0, 0), -1);
  }

  // Axis labels.
  cv::putText(
      plot, "Frame Index (X - axis)",
      cv::Point(leftMargin + plotW / 3, topMargin + plotH + bottomMargin - 5),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  cv::putText(plot, "Error (Y - axis)", cv::Point(5, topMargin - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  return plot;
}

cv::Mat
CalibrationVisualizer::drawDistortionGrid(const CameraCalibrationResult &res,
                                          int gridSize,
                                          cv::Size imageSize) const {
  cv::Mat output(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  // Generate ideal grid points
  std::vector<cv::Point3f> idealPoints;
  idealPoints.reserve((gridSize + 1) * (gridSize + 1));
  for (int i = 0; i <= gridSize; i++) {
    float y = i * (float)imageSize.height / gridSize;
    for (int j = 0; j <= gridSize; j++) {
      float x = j * (float)imageSize.width / gridSize;
      idealPoints.push_back(cv::Point3f(x, y, 0.f));
    }
  }

  std::vector<cv::Point2f> distorted;
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

  cv::projectPoints(idealPoints, rvec, tvec, res.cameraMatrix, res.distCoeffs,
                    distorted);

  for (int i = 0; i <= gridSize; i++) {
    int rowIdx = i * (gridSize + 1);
    for (int j = 0; j < gridSize; j++) {
      cv::Point2f p1 = distorted[rowIdx + j];
      cv::Point2f p2 = distorted[rowIdx + j + 1];
      cv::line(output, p1, p2, cv::Scalar(0, 0, 0), 1);
    }
  }
  for (int j = 0; j <= gridSize; j++) {
    for (int i = 0; i < gridSize; i++) {
      int idx1 = i * (gridSize + 1) + j;
      int idx2 = (i + 1) * (gridSize + 1) + j;
      cv::Point2f p1 = distorted[idx1];
      cv::Point2f p2 = distorted[idx2];
      cv::line(output, p1, p2, cv::Scalar(0, 0, 0), 1);
    }
  }

  // Generate a unique filename using the current time to ensure each image is
  // saved uniquely.
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
    const std::string &windowName) const {
#ifdef HAVE_OPENCV_VIZ
  // Create a 3D visualization window
  cv::viz::Viz3d vizWin(windowName);

  // Set background to black for better contrast
  vizWin.setBackgroundColor(cv::viz::Color::black());

  // Show a global coordinate system at the origin (scale = 0.1)
  vizWin.showWidget("GlobalCoord", cv::viz::WCoordinateSystem(0.1));

  // Register keyboard controls for zoom in ('+') and zoom out ('-')
  vizWin.registerKeyboardCallback(
      [](const cv::viz::KeyboardEvent &event, void *cookie) {
        cv::viz::Viz3d *win = static_cast<cv::viz::Viz3d *>(cookie);
        if (event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN) {
          // Get current camera position
          cv::Affine3d camPose = win->getViewerPose();
          cv::Vec3d pos = camPose.translation();

          // Zoom in: move camera 10% closer
          if (event.code == '+') {
            pos *= 0.9;
          }
          // Zoom out: move camera 10% farther
          else if (event.code == '-') {
            pos *= 1.1;
          }

          camPose.translation() = pos;
          win->setViewerPose(camPose);
        }
      },
      &vizWin);

  // Visualize each camera
  for (const auto &kv : realPoses) {
    int camId = kv.first;
    const cv::Mat &R = kv.second.first;
    const cv::Mat &t = kv.second.second;

    // Compute inverse pose for proper display
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;

    cv::Matx33d R33(R_inv);
    cv::Vec3d tVec(t_inv);
    cv::Affine3d pose(R33, tVec);

    // Display a local coordinate system for the camera
    std::string widgetName = "Camera_" + std::to_string(camId);
    cv::viz::WCameraPosition camCoord(0.05); // Small axis
    vizWin.showWidget(widgetName + "_coord", camCoord, pose);

    // If intrinsics are available, also draw the camera frustum
    auto it = calibResults.find(camId);
    if (it != calibResults.end()) {
      const CameraCalibrationResult &cres = it->second;
      CV_Assert(cres.cameraMatrix.type() == CV_64F &&
                cres.cameraMatrix.size() == cv::Size(3, 3));
      cv::Matx33d K(cres.cameraMatrix);
      cv::viz::WCameraPosition frustum(K, 0.05, cv::viz::Color::yellow());
      vizWin.showWidget(widgetName + "_frustum", frustum, pose);
    }

    // Add a 3D text label for the camera
    cv::viz::WText3D text3d("cam " + std::to_string(camId),
                            cv::Point3d(0, 0, 0.03), 0.02, true,
                            cv::viz::Color::white());
    vizWin.showWidget(widgetName + "_label", text3d, pose);
  }

  // Launch the interactive 3D viewer
  vizWin.spin();

#else
  (void)calibResults;
  (void)realPoses;
  (void)windowName;
  std::cout << "[Warning] OpenCV was built without the Viz module; 3D "
               "visualization is not available.\n";
#endif
}

} // namespace ccalib
} // namespace cv
