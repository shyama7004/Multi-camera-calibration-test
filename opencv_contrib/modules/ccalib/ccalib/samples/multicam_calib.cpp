#include "opencv2/calib3d.hpp"
#include "opencv2/ccalib/calibration_utils.hpp"
#include "opencv2/ccalib/fiducial_detectors.hpp"
#include "opencv2/ccalib/multicam_calibrator.hpp"
#include "opencv2/ccalib/visualizer.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <map>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ccalib;

static void help() {
  cout << "\nMulti-camera calibration using MultiCameraCalibrator and "
          "CalibrationVisualizer."
       << endl;
  cout << "Usage:" << endl;
  cout << "    multicam_calibration <multi_camera_config.yaml>" << endl;
  cout << "\nThe YAML should look like:" << endl;
  cout << "cameras:" << endl;
  cout << "  cam1:" << endl;
  cout << "    camera_id: 1" << endl;
  cout << "    pattern_type: chessboard" << endl;
  cout << "    pattern_size: [9, 6]" << endl;
  cout << "    square_size: 0.025" << endl;
  cout << "    image_paths:" << endl;
  cout << "      - \"images/cam1/img1.jpg\"" << endl;
  cout << "      - \"images/cam1/img2.jpg\"" << endl;
  cout << endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <multicam_config.yaml>" << endl;
    return -1;
  }

  // Define output directories.
  string resultsDir = "calibration_results";
  string plotDir = resultsDir + "/plot";

  // Create the directories if they do not exist.
  if (!cv::utils::fs::exists(resultsDir)) {
    cv::utils::fs::createDirectories(resultsDir);
  }
  if (!cv::utils::fs::exists(plotDir)) {
    cv::utils::fs::createDirectories(plotDir);
  }

  // Create an instance of the MultiCameraCalibrator.
  MultiCameraCalibrator calibrator;
  // Enable display (set to false if running in a headless environment).
  calibrator.setDisplay(true);

  // Load multi-camera configuration from a single YAML file.
  // The YAML file must contain a "cameras:" map with each camera's settings.
  calibrator.loadMultiCameraConfig(argv[1]);

  // Run calibration for each camera.
  // The boolean parameter indicates that relative poses (extrinsics) should be
  // computed.
  calibrator.calibrate(true);

  // Save the calibration results (including the YAML file) directly into
  // resultsDir.
  string outputFile = resultsDir + "/multicam_calibration_results.yaml";
  calibrator.saveResults(outputFile);
  cout << "Calibration results saved to " << outputFile << endl;

  // Generate a simple report.
  // Here we assume that generateReport saves plots into a "plot" subfolder
  // inside the given directory. (If generateReport does not automatically use
  // the plot subfolder, you can modify its implementation accordingly.)
  calibrator.generateReport(resultsDir);
  cout << "Calibration report generated in " << resultsDir << endl;

  return 0;
}
