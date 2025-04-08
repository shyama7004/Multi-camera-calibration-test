# Multi-Camera Calibration Test

This project is a tool for testing and visualizing multi-camera calibration. It was developed as part of a Google Summer of Code initiative to enhance and streamline the calibration process for multi-camera setups. By using this tool, developers can easily test calibration pipelines, visualize camera poses in 3D, and analyze error distributions.

A short demo video is available below, click on it to get a quick overview of the project:

<div align="center">
  <a href="https://www.youtube.com/watch?v=YzzY50zVvOg">
    <img src="https://github.com/shyama7004/Multi-camera-calibration-test/blob/main/images/Multi-Camera%20Calibration%20test.png" alt="Watch the demo" style="width:400px; height:300px;">
  </a>
</div>


## Requirements

Before you begin, ensure that you have the following:
- [OpenCV](https://github.com/opencv/opencv) repository.
- [OpenCV Contrib](https://github.com/opencv/opencv_contrib) repository (provides extra modules).
- My C++ calibration implementation (copy the necessay files from my repo into `ccalib` module of your `opencv_contrib` repo).
- Curated raw images are available on [Kaggle](https://www.kaggle.com/datasets/kalpitnathan/multi-camera-calibration-test-dataset), along with additional public datasets. (These images are clear, low-noise, and synchronized.)

---

## Build Instructions

### 1. Clone Required Repositories

Clone the necessary repositories to your local machine:
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

### 2. Prepare the OpenCV Build

**Integrate Calibration Code:**
Copy the required calibration source files from the `ccalib` implementation into the appropriate directories within the OpenCV Contrib tree.

Create a build directory within the OpenCV source folder and run CMake. Replace the placeholder paths with the actual paths on your system:

```bash
cmake \
  -DOPENCV_EXTRA_MODULES_PATH=<path-to-opencv_contrib>/modules \
  -DOPENCV_TEST_DATA_PATH=<path-to-opencv_extra_testdata> \
  -DWITH_OPENMP=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_opencv_ccalib=ON \
  -DBUILD_opencv_aruco=ON \
  -DBUILD_opencv_viz=ON \
  -DWITH_VTK=ON \
  -DBUILD_opencv_samples=ON \
  -DBUILD_opencv_python3=ON \
  ..
```

Then, compile the project (for example, using make):
```bash
make -j$(nproc)
```

---

## Running the Sample Code

### 1. Compile the Calibration Tool

After building OpenCV, compile the [sample calibration](https://github.com/shyama7004/Multi-camera-calibration-test/blob/main/opencv_contrib/modules/ccalib/ccalib/samples/multicam_calib.cpp) code using your favorite compiler. For example, using GCC with C++17:
```bash
g++ multicam_calib.cpp -o multicam_calib \
    -I<path-to-opencv_include> \
    -L<path-to-opencv_lib> \
    -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
    -lopencv_calib3d -lopencv_ccalib -lopencv_aruco -lopencv_viz \
    -std=c++17 -Wl,-rpath,<path-to-opencv_lib>
```
Replace `<path-to-opencv_include>` and `<path-to-opencv_lib>` with the actual directories from your OpenCV installation.

### 2. Configure and Run the Tool

- **Dataset:**
  Select a sample YAML configuration file from my  curated dataset availabe on [kaggle](). This file should specify the image paths and calibration settings.

- **Update YAML File:**
  Use the provided sample YAML configuration file from my [Kaggle dataset](). This YAML file specifies the image paths and calibration settings for each fiducial pattern. You can also update the YAML file to point to any additional public datasets you prefer.

> Note: Due to time constraints, the fisheye implementation is not available yet. This project is in its initial stage and will be refined as development continues.

- **Execute Calibration:**
  Run the compiled tool with the YAML file as an argument:
  ```bash
  ./multicam_calib <path-to-yaml-config>
  ```
  For example:
  ```bash
  ./multicam_calib /path/to/dataset/multicam_chessboard.yaml
  ```

Once executed, the tool will:

- Display a 3D visualization window showing camera positions and orientations. See the example below (a GIF).
- Generate error histograms and plots to help you assess calibration accuracy.

To close the 3D window, press `q`. After reviewing the plots, press `q` again to save the results in the `calibration_results` directory.

### 3D Visualization of Camera Positions

Below is a GIF demonstrating the 3D visualization of camera positions and orientations, providing an interactive view of the calibration setup.

![3d_viz](https://github.com/shyama7004/Multi-camera-calibration-test/blob/main/images/3d_viz.gif)

Below are side-by-side examples of an error histogram and a reprojection errors plot:

<table>
  <tr>
    <td><img src="https://github.com/shyama7004/Multi-camera-calibration-test/blob/main/images/cam1_histogram.png" width="400"></td>
    <td><img src="https://github.com/shyama7004/Multi-camera-calibration-test/blob/main/images/cam1_reprojErrors.png" width="400"></td>
  </tr>
  <tr>
    <td align="center">Error Histogram</td>
    <td align="center">Reprojection Errors Plot</td>
  </tr>
</table>

---

## Project Structure

<img src = https://github.com/shyama7004/Multi-camera-calibration-test/blob/main/images/directory_structure.png>

> Note: Due to time constraints, only a limited set of tasks has been implemented so far, but additional features will be added as the project progresses.

```
Thanks for testing the code.
```
