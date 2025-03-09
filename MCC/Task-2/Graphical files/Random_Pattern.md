
<details>
<summary> Symmetric Circles Grid </summary>

```py

import cv2 as cv
import numpy as np
import glob
import os

# ---------------- User Settings ----------------
# For your printed asymmetric circles:
#   - Your generated board has 7 rows and 10 columns.
#   - In the generated image, centers are computed as:
#         center_x = j * spacing_x + (i % 2) * (spacing_x/2)
#         center_y = i * spacing_y
#   To match the OpenCV sample formula for an asymmetric circles grid:
#         x = (2*j + (i % 2)) * squareSize, y = i * squareSize
#   Set squareSize such that squareSize = spacing_x/2. For spacing_x = 100, we choose squareSize = 50.
pattern_size = (6, 7)  # (columns, rows): here 7 columns, 10 rows
use_asymmetric = True   # Use asymmetric grid detection.
invert_image = False    # Set to True if your pattern is white dots on a dark background.

# Real-world spacing: use the squareSize from the calibration target.
squareSize = 30.0  # in your chosen units (e.g., millimeters)

# Folder with calibration images (update this path)
image_dir = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/camera2 mac m1 symmetric circles"
image_extension = "*.png"

# Directory to save calibration results
save_dir = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/camera2 mac m1/res"
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, "calibration_result.yaml")

# ---------------- Prepare Object Points ----------------
# For an asymmetric circles grid, the tutorial recommends:
#   for i in 0 .. (rows-1):
#       for j in 0 .. (cols-1):
#           x = (2*j + (i % 2)) * squareSize, y = i * squareSize, z = 0
# Note: This formulation matches our printed grid if squareSize = spacing_x/2.
objp = []
rows_obj = pattern_size[1]
cols_obj = pattern_size[0]
for i in range(rows_obj):
    for j in range(cols_obj):
        if use_asymmetric:
            x = (2 * j + (i % 2)) * squareSize
        else:
            x = j * squareSize
        y = i * squareSize
        objp.append([x, y, 0])
objp = np.array(objp, dtype=np.float32)

# ---------------- Setup Custom Blob Detector ----------------
# These parameters may need to be tuned for your target.
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 300      # Adjust if your circles appear too small in the image
params.maxArea = 3000    # Adjust if circles are large
params.filterByCircularity = True
params.minCircularity = 0.2
params.filterByInertia = False
params.filterByConvexity = False

detector = cv.SimpleBlobDetector_create(params)

# ---------------- Load Calibration Images ----------------
images = glob.glob(os.path.join(image_dir, image_extension))
print(f"Found {len(images)} calibration images.")

# Containers for object points and image points
object_points = []  # 3D points in real-world space (will be the same for all images)
image_points = []   # 2D points in image plane

# ---------------- Process Each Image ----------------
for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Could not load image: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if invert_image:
        gray = 255 - gray

    # (Optional) Show detected blobs for diagnostic purposes.
    keypoints = detector.detect(gray)
    debug_img = cv.drawKeypoints(gray, keypoints, None, (0, 0, 255),
                                 cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Blobs Debug", debug_img)
    cv.waitKey(200)  # brief pause for diagnosis

    # Set flag for grid type.
    flag = cv.CALIB_CB_ASYMMETRIC_GRID if use_asymmetric else cv.CALIB_CB_SYMMETRIC_GRID

    ret, centers = cv.findCirclesGrid(gray, pattern_size, flags=flag, blobDetector=detector)
    if ret:
        # Improve accuracy with cornerSubPix.
        centers_refined = cv.cornerSubPix(gray, centers, (11, 11), (-1, -1),
                                          (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(objp)
        image_points.append(centers_refined)
        cv.drawChessboardCorners(img, pattern_size, centers_refined, ret)
        cv.imshow("Detected Pattern", img)
        cv.waitKey(500)
    else:
        print(f"Pattern not found in image: {fname}")

cv.destroyAllWindows()

# ---------------- Camera Calibration ----------------
if len(object_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )
    print("Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # ---------------- Compute Reprojection Error ----------------
    total_error = 0
    total_points = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(image_points[i], imgpoints2, cv.NORM_L2)
        n = len(object_points[i])
        total_error += error**2
        total_points += n
    mean_error = np.sqrt(total_error / total_points)
    print(f"Average reprojection error: {mean_error}")

    # ---------------- Save Calibration Results ----------------
    fs_write = cv.FileStorage(save_file, cv.FILE_STORAGE_WRITE)
    fs_write.write("camera_matrix", camera_matrix)
    fs_write.write("distortion_coefficients", dist_coeffs)
    fs_write.write("reprojection_error", mean_error)
    fs_write.release()
    print(f"Calibration results saved to {save_file}")

    # ---------------- (Optional) Undistort a Sample Image ----------------
    # Compute undistortion maps.
    newCamMat, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, gray.shape[::-1], 1, gray.shape[::-1])
    map1, map2 = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newCamMat, gray.shape[::-1], cv.CV_16SC2)
    # Show one undistorted image.
    sample_img = cv.imread(images[0])
    undistorted = cv.remap(sample_img, map1, map2, interpolation=cv.INTER_LINEAR)
    cv.imshow("Undistorted Image", undistorted)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No valid calibration images were found. Calibration aborted.")


```

</details>


<details>
<summary> Asymmetric Circles Grid </summary>

```py

import cv2 as cv
import numpy as np
import glob
import os

# ---------------- User Settings ----------------
# For your printed asymmetric circles:
#   - Your generated board has 7 rows and 10 columns.
#   - In the generated image, centers are computed as:
#         center_x = j * spacing_x + (i % 2) * (spacing_x/2)
#         center_y = i * spacing_y
#   To match the OpenCV sample formula for an asymmetric circles grid:
#         x = (2*j + (i % 2)) * squareSize, y = i * squareSize
#   Set squareSize such that squareSize = spacing_x/2. For spacing_x = 100, we choose squareSize = 50.
pattern_size = (7, 10)  # (columns, rows): here 7 columns, 10 rows
use_asymmetric = True   # Use asymmetric grid detection.
invert_image = False    # Set to True if your pattern is white dots on a dark background.

# Real-world spacing: use the squareSize from the calibration target.
squareSize = 50.0  # in your chosen units (e.g., millimeters)

# Folder with calibration images (update this path)
image_dir = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/camera2 mac m1"
image_extension = "*.png"

# Directory to save calibration results
save_dir = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/camera2 mac m1/res"
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, "calibration_result.yaml")

# ---------------- Prepare Object Points ----------------
# For an asymmetric circles grid, the tutorial recommends:
#   for i in 0 .. (rows-1):
#       for j in 0 .. (cols-1):
#           x = (2*j + (i % 2)) * squareSize, y = i * squareSize, z = 0
# Note: This formulation matches our printed grid if squareSize = spacing_x/2.
objp = []
rows_obj = pattern_size[1]
cols_obj = pattern_size[0]
for i in range(rows_obj):
    for j in range(cols_obj):
        if use_asymmetric:
            x = (2 * j + (i % 2)) * squareSize
        else:
            x = j * squareSize
        y = i * squareSize
        objp.append([x, y, 0])
objp = np.array(objp, dtype=np.float32)

# ---------------- Setup Custom Blob Detector ----------------
# These parameters may need to be tuned for your target.
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50      # Adjust if your circles appear too small in the image
params.maxArea = 5000    # Adjust if circles are large
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByInertia = False
params.filterByConvexity = False

detector = cv.SimpleBlobDetector_create(params)

# ---------------- Load Calibration Images ----------------
images = glob.glob(os.path.join(image_dir, image_extension))
print(f"Found {len(images)} calibration images.")

# Containers for object points and image points
object_points = []  # 3D points in real-world space (will be the same for all images)
image_points = []   # 2D points in image plane

# ---------------- Process Each Image ----------------
for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Could not load image: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if invert_image:
        gray = 255 - gray

    # (Optional) Show detected blobs for diagnostic purposes.
    keypoints = detector.detect(gray)
    debug_img = cv.drawKeypoints(gray, keypoints, None, (0, 0, 255),
                                 cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Blobs Debug", debug_img)
    cv.waitKey(200)  # brief pause for diagnosis

    # Set flag for grid type.
    flag = cv.CALIB_CB_ASYMMETRIC_GRID if use_asymmetric else cv.CALIB_CB_SYMMETRIC_GRID

    ret, centers = cv.findCirclesGrid(gray, pattern_size, flags=flag, blobDetector=detector)
    if ret:
        # Improve accuracy with cornerSubPix.
        centers_refined = cv.cornerSubPix(gray, centers, (11, 11), (-1, -1),
                                          (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(objp)
        image_points.append(centers_refined)
        cv.drawChessboardCorners(img, pattern_size, centers_refined, ret)
        cv.imshow("Detected Pattern", img)
        cv.waitKey(500)
    else:
        print(f"Pattern not found in image: {fname}")

cv.destroyAllWindows()

# ---------------- Camera Calibration ----------------
if len(object_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )
    print("Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # ---------------- Compute Reprojection Error ----------------
    total_error = 0
    total_points = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(image_points[i], imgpoints2, cv.NORM_L2)
        n = len(object_points[i])
        total_error += error**2
        total_points += n
    mean_error = np.sqrt(total_error / total_points)
    print(f"Average reprojection error: {mean_error}")

    # ---------------- Save Calibration Results ----------------
    fs_write = cv.FileStorage(save_file, cv.FILE_STORAGE_WRITE)
    fs_write.write("camera_matrix", camera_matrix)
    fs_write.write("distortion_coefficients", dist_coeffs)
    fs_write.write("reprojection_error", mean_error)
    fs_write.release()
    print(f"Calibration results saved to {save_file}")

    # ---------------- (Optional) Undistort a Sample Image ----------------
    # Compute undistortion maps.
    newCamMat, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, gray.shape[::-1], 1, gray.shape[::-1])
    map1, map2 = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newCamMat, gray.shape[::-1], cv.CV_16SC2)
    # Show one undistorted image.
    sample_img = cv.imread(images[0])
    undistorted = cv.remap(sample_img, map1, map2, interpolation=cv.INTER_LINEAR)
    cv.imshow("Undistorted Image", undistorted)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No valid calibration images were found. Calibration aborted.")


```
</details>

