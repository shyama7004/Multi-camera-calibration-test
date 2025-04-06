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

namespace {

// helper function to draw common axes for plots
static void drawAxes(cv::Mat &plot, int leftMargin, int topMargin, int plotW,
                     int plotH, const std::string &xLabel,
                     const std::string &yLabel, int bottomMargin = 0) {
  // Draw vertical axis
  cv::line(plot, cv::Point(leftMargin, topMargin),
           cv::Point(leftMargin, topMargin + plotH), vizcolors::AXIS_COLOR, 2);
  // Draw horizontal axis
  cv::line(plot, cv::Point(leftMargin, topMargin + plotH),
           cv::Point(leftMargin + plotW, topMargin + plotH),
           vizcolors::AXIS_COLOR, 2);
  // Draw labels
  cv::putText(
      plot, xLabel,
      cv::Point(leftMargin + plotW / 3,
                topMargin + plotH + (bottomMargin > 0 ? bottomMargin : 35)),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, vizcolors::AXIS_COLOR, 1);
  cv::putText(plot, yLabel, cv::Point(leftMargin - 45, topMargin - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, vizcolors::AXIS_COLOR, 1);
}

} // namespace

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

    // Draw detected corner in green using constant
    cv::circle(overlay, dPt, 3, vizcolors::DETECTED_COLOR, -1);
    // Draw a red line from the reprojected point to the detected point
    cv::line(overlay, rPt, dPt, cv::Scalar(0, 0, 255), 2);
    // Draw reprojected corner in yellowish color
    cv::circle(overlay, rPt, 3, vizcolors::REPROJECTED_COLOR, -1);
  }
  return overlay;
}

cv::Mat
CalibrationVisualizer::plotErrorHistogram(const std::vector<double> &errors,
                                          int histSize, int histWidth,
                                          int histHeight) const {
  if (errors.empty()) {
    return cv::Mat(histHeight, histWidth, CV_8UC3, vizcolors::PLOT_BG);
  }

  double minVal, maxVal;
  cv::minMaxLoc(errors, &minVal, &maxVal);
  double range = maxVal - minVal;
  if (range < 1e-12)
    range = 1e-12;

  // Count errors into bins.
  std::vector<int> bins(histSize, 0);
  double binWidth = range / static_cast<double>(histSize);
  for (double e : errors) {
    int idx = static_cast<int>((e - minVal) / binWidth);
    if (idx >= histSize)
      idx = histSize - 1;
    bins[idx]++;
  }

  int maxCount = 0;
  for (int c : bins)
    maxCount = std::max(maxCount, c);

  // Define margins.
  int leftMargin = 50;
  int bottomMargin = 50;
  int topMargin = 30;
  int rightMargin = 30;

  int plotW = histWidth - leftMargin - rightMargin;
  int plotH = histHeight - topMargin - bottomMargin;

  cv::Mat hist(histHeight, histWidth, CV_8UC3, vizcolors::PLOT_BG);
  double scale = static_cast<double>(plotH) / maxCount;
  int binPixWidth = cvRound(static_cast<double>(plotW) / histSize);

  // Draw histogram bars.
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

  // Draw axes using helper function.
  drawAxes(hist, leftMargin, topMargin, plotW, plotH,
           "Reprojection Error (X - axis)", "Frequency (Y - axis)",
           bottomMargin);

  // X-axis tick labels.
  int xTicks = 5;
  for (int i = 0; i <= xTicks; i++) {
    double tickValue = minVal + i * (range / xTicks);
    int x = leftMargin + cvRound((tickValue - minVal) / range * plotW);
    cv::line(hist, cv::Point(x, topMargin + plotH),
             cv::Point(x, topMargin + plotH + 5), vizcolors::AXIS_COLOR, 1);
    cv::putText(hist, cv::format("%.2f", tickValue),
                cv::Point(x - 15, topMargin + plotH + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, vizcolors::AXIS_COLOR, 1);
  }

  // Y-axis tick labels.
  int yTicks = 5;
  for (int i = 0; i <= yTicks; i++) {
    int tickVal = cvRound((maxCount / (double)yTicks) * i);
    int y = topMargin + plotH - cvRound(tickVal * scale);
    cv::line(hist, cv::Point(leftMargin - 5, y), cv::Point(leftMargin, y),
             vizcolors::AXIS_COLOR, 1);
    cv::putText(hist, std::to_string(tickVal),
                cv::Point(leftMargin - 45, y + 5), cv::FONT_HERSHEY_SIMPLEX,
                0.45, vizcolors::AXIS_COLOR, 1);
  }

  return hist;
}

cv::Mat
CalibrationVisualizer::plotReprojErrorsLine(const std::vector<double> &errors,
                                            int width, int height) const {
  cv::Mat plot(height, width, CV_8UC3, vizcolors::PLOT_BG);
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

  // Draw axes using helper function.
  drawAxes(plot, leftMargin, topMargin, plotW, plotH, "Frame Index (X - axis)",
           "Error (Y - axis)", bottomMargin);

  int n = static_cast<int>(errors.size());

  // X-axis tick labels.
  int xTicks = std::min(n, 5);
  for (int i = 0; i < xTicks; i++) {
    int x = leftMargin + static_cast<int>((double)i / (xTicks - 1) * plotW);
    cv::line(plot, cv::Point(x, topMargin + plotH),
             cv::Point(x, topMargin + plotH + 2), vizcolors::AXIS_COLOR, 1);
    cv::putText(plot, std::to_string(i),
                cv::Point(x - 10, topMargin + plotH + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, vizcolors::AXIS_COLOR, 1);
  }

  // Y-axis tick labels.
  int yTicks = 5;
  for (int i = 0; i <= yTicks; i++) {
    double value = minVal + i * (range / yTicks);
    int y = topMargin + plotH -
            static_cast<int>(((value - minVal) / range) * plotH);
    cv::line(plot, cv::Point(leftMargin - 5, y), cv::Point(leftMargin, y),
             vizcolors::AXIS_COLOR, 1);
    cv::putText(plot, cv::format("%.2f", value), cv::Point(5, y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, vizcolors::AXIS_COLOR, 1);
  }

  // Plot the error line using the defined error color.
  for (int i = 0; i < n - 1; i++) {
    double normY1 = (errors[i] - minVal) / range;
    double normY2 = (errors[i + 1] - minVal) / range;
    int x1 = leftMargin + static_cast<int>((double)i / (n - 1) * plotW);
    int x2 = leftMargin + static_cast<int>((double)(i + 1) / (n - 1) * plotW);
    int y1 = topMargin + plotH - static_cast<int>(normY1 * plotH);
    int y2 = topMargin + plotH - static_cast<int>(normY2 * plotH);
    cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), vizcolors::ERROR_COLOR,
             2);
    cv::circle(plot, cv::Point(x1, y1), 2, vizcolors::ERROR_COLOR, -1);
  }
  if (n > 1) {
    double normY = (errors.back() - minVal) / range;
    int xLast = leftMargin + plotW;
    int yLast = topMargin + plotH - static_cast<int>(normY * plotH);
    cv::circle(plot, cv::Point(xLast, yLast), 2, vizcolors::ERROR_COLOR, -1);
  }

  return plot;
}
cv::Mat
CalibrationVisualizer::drawDistortionGrid(const CameraCalibrationResult &res,
                                          int gridSize,
                                          cv::Size fullImageSize) const {
  // Extend the grid region by a fraction (e.g., 50% extra on each side)
  float offsetFrac = 0.5f;
  float offsetX = offsetFrac * fullImageSize.width;
  float offsetY = offsetFrac * fullImageSize.height;
  float extendedWidth = fullImageSize.width + 2 * offsetX;
  float extendedHeight = fullImageSize.height + 2 * offsetY;

  // Generate ideal grid points over the extended region.
  // The grid spans from (-offsetX, -offsetY) to (fullImageSize.width+offsetX,
  // fullImageSize.height+offsetY)
  std::vector<cv::Point3f> idealPoints;
  idealPoints.reserve((gridSize + 1) * (gridSize + 1));
  for (int i = 0; i <= gridSize; i++) {
    float y = -offsetY + (i / static_cast<float>(gridSize)) * extendedHeight;
    for (int j = 0; j <= gridSize; j++) {
      float x = -offsetX + (j / static_cast<float>(gridSize)) * extendedWidth;
      idealPoints.push_back(cv::Point3f(x, y, 0.f));
    }
  }

  // Project the ideal grid points using zero rotation/translation.
  std::vector<cv::Point2f> projected;
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::projectPoints(idealPoints, rvec, tvec, res.cameraMatrix, res.distCoeffs,
                    projected);

  // Compute the bounding rectangle of all valid projected points.
  float minX = std::numeric_limits<float>::max();
  float minY = std::numeric_limits<float>::max();
  float maxX = std::numeric_limits<float>::lowest();
  float maxY = std::numeric_limits<float>::lowest();
  for (const auto &pt : projected) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y))
      continue;
    minX = std::min(minX, pt.x);
    minY = std::min(minY, pt.y);
    maxX = std::max(maxX, pt.x);
    maxY = std::max(maxY, pt.y);
  }
  if (minX == std::numeric_limits<float>::max() ||
      minY == std::numeric_limits<float>::max()) {
    return cv::Mat(); // No valid points found.
  }
  float bboxWidth = maxX - minX;
  float bboxHeight = maxY - minY;

  // Compute scaling factor to map the bounding box into fullImageSize.
  float scaleX = fullImageSize.width / bboxWidth;
  float scaleY = fullImageSize.height / bboxHeight;
  float scale = std::min(scaleX, scaleY);
  float displayWidth = bboxWidth * scale;
  float displayHeight = bboxHeight * scale;
  // Compute offsets to center the grid.
  float offsetDisplayX = (fullImageSize.width - displayWidth) / 2.0f;
  float offsetDisplayY = (fullImageSize.height - displayHeight) / 2.0f;

  // Map projected points to display coordinates.
  std::vector<cv::Point2f> displayPoints;
  displayPoints.resize(projected.size());
  for (size_t i = 0; i < projected.size(); i++) {
    displayPoints[i].x = (projected[i].x - minX) * scale + offsetDisplayX;
    displayPoints[i].y = (projected[i].y - minY) * scale + offsetDisplayY;
  }

  // Create output image with the fixed fullImageSize.
  cv::Mat output(fullImageSize, CV_8UC3, vizcolors::PLOT_BG);

  // Draw horizontal grid lines.
  for (int i = 0; i <= gridSize; i++) {
    for (int j = 0; j < gridSize; j++) {
      cv::Point2f p1 = displayPoints[i * (gridSize + 1) + j];
      cv::Point2f p2 = displayPoints[i * (gridSize + 1) + j + 1];
      cv::line(output, p1, p2, cv::Scalar(0, 0, 0), 1);
    }
  }
  // Draw vertical grid lines.
  for (int j = 0; j <= gridSize; j++) {
    for (int i = 0; i < gridSize; i++) {
      cv::Point2f p1 = displayPoints[i * (gridSize + 1) + j];
      cv::Point2f p2 = displayPoints[(i + 1) * (gridSize + 1) + j];
      cv::line(output, p1, p2, cv::Scalar(0, 0, 0), 1);
    }
  }
  return output;
}

void CalibrationVisualizer::plotExtrinsics3D(
    const std::map<int, CameraCalibrationResult> &calibResults,
    int /*referenceCameraId*/,
    const std::map<int, std::pair<cv::Mat, cv::Mat>> &realPoses,
    const std::string &windowName) const {
#ifdef HAVE_OPENCV_VIZ
  // create a 3d visulisation window.
  cv::viz::Viz3d vizWin(windowName);
  // the background is set to black for better contrast.
  vizWin.setBackgroundColor(cv::viz::Color::black());

  vizWin.showWidget("GlobalCoord", cv::viz::WCoordinateSystem(0.1));

  // Visualize each camera.
  for (const auto &kv : realPoses) {
    int camId = kv.first;
    const cv::Mat &R = kv.second.first;
    const cv::Mat &t = kv.second.second;

    // compute the inverse pose for proper display
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;
    cv::Matx33d R33(R_inv);
    cv::Vec3d tVec(t_inv);
    cv::Affine3d pose(R33, tVec);

    std::string widgetName = "Camera_" + std::to_string(camId);
    cv::viz::WCameraPosition camCoord(0.05);
    vizWin.showWidget(widgetName + "_coord", camCoord, pose);

    // If intrinsics are available, also draw the camera frustum.
    auto it = calibResults.find(camId);
    if (it != calibResults.end()) {
      const CameraCalibrationResult &cres = it->second;
      CV_Assert(cres.cameraMatrix.type() == CV_64F &&
                cres.cameraMatrix.size() == cv::Size(3, 3));
      cv::Matx33d K(cres.cameraMatrix);
      cv::viz::WCameraPosition frustum(K, 0.05, cv::viz::Color::yellow());
      vizWin.showWidget(widgetName + "_frustum", frustum, pose);
    }

    // Add a 3D text label for the camera.
    cv::viz::WText3D text3d("cam " + std::to_string(camId),
                            cv::Point3d(0, 0, 0.03), 0.02, true,
                            cv::viz::Color::white());
    vizWin.showWidget(widgetName + "_label", text3d, pose);
  }

  // Launch the interactive 3D viewer.
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
