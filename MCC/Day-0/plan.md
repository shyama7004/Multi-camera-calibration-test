Link of Camera calibration with square chessboard : [Click Me](https://docs.opencv.org/4.x/dc/d43/tutorial_camera_calibration_square_chess.html)

<details><summary> related file </summary>

Here is the Markdown version of your document:

```md
# OpenCV: Camera calibration with square chessboard

## Table of Contents
- Pose estimation
- Camera calibration with square chessboard

Prev Tutorial: Create calibration pattern  
Next Tutorial: Camera calibration With OpenCV  

**Original author:** Victor Eruhimov  
**Compatibility:** OpenCV >= 4.0  

The goal of this tutorial is to learn how to calibrate a camera given a set of chessboard images.

**Test data:** use images in your `data/chess` folder.

Compile OpenCV with samples by setting `BUILD_EXAMPLES` to `ON` in CMake configuration.  
Go to the `bin` folder and use `imagelist_creator` to create an XML/YAML list of your images.  
Then, run the calibration sample to get camera parameters. Use a square size equal to 3cm.

---

## Pose estimation

Now, let us write code that detects a chessboard in an image and finds its distance from the camera.  
You can apply this method to any object with known 3D geometry, which you detect in an image.

**Test data:** use `chess_test*.jpg` images from your data folder.

1. Create an empty console project. Load a test image:

   ```cpp
   Mat img = imread(argv[1], IMREAD_GRAYSCALE);
   ```

2. Detect a chessboard in this image using the `findChessboardCorners` function:

   ```cpp
   bool found = findChessboardCorners(img, boardSize, ptvec, CALIB_CB_ADAPTIVE_THRESH);
   ```

3. Write a function that generates a `vector<Point3f>` array of 3D coordinates of a chessboard in any coordinate system.  
   For simplicity, let us choose a system such that one of the chessboard corners is at the origin, and the board is in the plane `z = 0`.

4. Read camera parameters from an XML/YAML file:

   ```cpp
   FileStorage fs(filename, FileStorage::READ);
   Mat intrinsics, distortion;
   fs["camera_matrix"] >> intrinsics;
   fs["distortion_coefficients"] >> distortion;
   ```

5. Now we are ready to find a chessboard pose by running `solvePnP`:

   ```cpp
   solvePnP(Mat(boardPoints), Mat(foundBoardCorners), cameraMatrix,
                     distCoeffs, rvec, tvec, false);
   ```

6. Calculate reprojection error like it is done in the calibration sample  
   (see `opencv/samples/cpp/calibration.cpp`, function `computeReprojectionErrors`).

**Question:** How would you calculate the distance from the camera origin to any one of the corners?  
**Answer:** As our image lies in 3D space, firstly we would calculate the relative camera pose.  
This would give us 3D to 2D correspondences.  
Next, we can apply a simple L2 norm to calculate the distance between any point (end point for corners).

---

**Generated on:** Wed Mar 5, 2025, 23:07:02 for OpenCV by 1.12.0  

### References:
- [Camera calibration with square chessboard](https://docs.opencv.org/4.x/dc/d43/tutorial_camera_calibration_square_chess.html)
- [Camera calibration pattern](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
- [Camera calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
- [Doxygen](https://www.doxygen.org/index.html)

</details>

Link of Calibration with ArUco and ChArUco : [Click Me](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)

<details><summary>Sample file for generating grids and stuffs </summary>


```py
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_checkerboard(rows, cols, square_size=50):
    """Generates a checkerboard pattern."""
    board_size = (rows * square_size, cols * square_size)
    checkerboard = np.ones((board_size[1], board_size[0]), dtype=np.uint8) * 255
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                cv2.rectangle(checkerboard, 
                              (i * square_size, j * square_size), 
                              ((i + 1) * square_size, (j + 1) * square_size), 
                              (0, 0, 0), -1)
    
    return checkerboard

def create_circle_grid(rows, cols, radius=15, spacing=50):
    """Generates a circle grid pattern."""
    height = rows * spacing
    width = cols * spacing
    circle_grid = np.ones((height, width), dtype=np.uint8) * 255
    
    for i in range(rows):
        for j in range(cols):
            center = (j * spacing + spacing // 2, i * spacing + spacing // 2)
            cv2.circle(circle_grid, center, radius, (0, 0, 0), -1)
    
    return circle_grid

def display_pattern(pattern, title):
    """Displays a pattern using Matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(pattern, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Create and display patterns
checkerboard = create_checkerboard(9, 6, 50)
display_pattern(checkerboard, "Checkerboard Pattern")

circle_grid = create_circle_grid(7, 5, 15, 50)
display_pattern(circle_grid, "Circle Grid Pattern")
```
</details>
