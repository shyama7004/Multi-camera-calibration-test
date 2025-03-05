### We are looking for a student to curate best of class calibration data(Task-1)

The table below summarizes recommended image requirements for common OpenCV calibration patterns. Values like “~10–20 images” are guidelines, not strict rules—more may be needed for wide-angle or fisheye lenses or higher accuracy.

| **Pattern Type**         | **Recommended # of Images**        | **Coverage & Variety**                                                                                                                                                                | **Usage (Internal/External Params)**                               | **Unique Orientation?** | **Pros**                                                                                                                                                    | **Cons**                                                                                                                                |
|--------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| **Checkerboard**         | ~10–20 per camera-lens setup        | - Capture images from **different angles** (tilts, rotations).<br/>- Vary **distances** (close-up vs. far).<br/>- Ensure **pattern corners** appear in multiple regions of the image. | Primarily **internal** (focal length, distortion). Not well-suited for measuring external orientation due to rotational ambiguity around the optical axis. | No                        | - **Widely used**, easy to print.<br/>- Robust corner detection in good lighting.<br/>- Well-documented in OpenCV tutorials.           | - Orientation ambiguity (rotations can’t be distinguished).<br/>- Not suitable for external parameters (like absolute orientation).     |
| **Circle Grid**          | ~10–20 per camera-lens setup        | - Similar approach: multiple angles, distances.<br/>- Make sure **dots** are clearly visible and well-lit.                                                                             | Primarily **internal** (focal length, distortion). Also not ideal for external parameters (same orientation ambiguity as checkerboard).              | No                        | - Potentially **more accurate** centroid detection, especially if the board is tilted.<br/>- Often more robust than checkerboard in angled shots. | - Orientation ambiguity (no unique way to track rotation around optical axis).<br/>- Slightly more complex detection than checkerboard. |
| **Asymmetric Circle Grid** | ~10–20 per camera-lens setup      | - Vary angles and distances.<br/>- The asymmetry allows the algorithm to distinguish pattern orientation.                                                                              | **Both internal and external** (can measure orientation since the pattern’s rotation is uniquely determined).                                    | Yes                       | - Eliminates orientation ambiguity.<br/>- Good for measuring **both** camera intrinsic and extrinsic parameters.                        | - Requires a **custom asymmetric** pattern, which may be harder to print or generate.                                                   |
| **ChArUco Pattern**      | ~10–20 per camera-lens setup        | - Capture images from various angles/distances.<br/>- **Partial occlusions** are okay, as each ArUco marker is uniquely identified.                                                    | Suitable for **internal** and **external** parameters.<br/>Also excellent for **AR** or **robotics** where markers might be partially hidden.      | Yes                       | - Each square is uniquely identified, so occlusions are less problematic.<br/>- Measures internal/external parameters + lens distortion accurately. | - More complex to generate and detect.<br/>- Requires printing ArUco markers and using dedicated detection routines.                     |

<details> <summary> Additional Notes </summary>

1. **Minimum Number of Images:**  
   - **10–20** images is a general rule of thumb for a standard lens. For **wide-angle** or **fish-eye** lenses, you may need **20–30 or more** to accurately capture distortion.

2. **Variety & Coverage:**  
   - Ensure you vary **tilt**, **roll**, **distance**, and **lighting** conditions.  
   - Capture the pattern in **all corners** and the **center** of the frame.  
   - The goal is to see how the camera behaves under different parts of the lens and different angles.

3. **Print Quality & Pattern Size:**  
   - High-contrast printing on non-reflective paper is recommended.  
   - The size of the pattern (number of squares/dots) can influence detection accuracy.

4. **Data Organization:**  
   - Label each image set with camera type, lens info, pattern type, and any special conditions (e.g., “low light,” “outdoor,” etc.).  
   - This organization helps when you run automated test scripts later.

5. **Combining with Public Datasets:**  
   - You can **mix** self-collected data with **public datasets** (e.g., KITTI, TUM, EuRoC) to broaden coverage.  
   - Just ensure you understand how those datasets were captured (e.g., fiducial used, camera specs).

By following these guidelines, you’ll gather enough **high-quality, diverse** calibration images to thoroughly test and compare the accuracy and reliability of each calibration pattern.
</details>


### **Day 1: Understanding Calibration Requirements**
- Review OpenCV’s calibration documentation for single and multi-camera setups.
- Identify supported calibration patterns:  
  - **Chessboard**
  - **Circle Grid (symmetric/asymmetric)**
  - **ArUco**
  - **ChArUco**
  - **Random Dot**
- List the required data (images) for each pattern.
- Review OpenCV’s testing guidelines, including:
  - Storage policies for test data.
  - Usage of `opencv_extra` test data repository.
 
#### I think some images ar eto be stored in opencv and majority of images in opencv_extra.

---

### **Day 2: Researching Calibration Datasets**
- Identify open, copyright-free datasets or sample images.
- Explore sources such as:
  - OpenCV’s `opencv_extra` repository.
  - Public domain datasets used in OpenCV examples.
- Evaluate existing datasets for reuse.
- Search for multi-camera calibration datasets (stereo/multi-view).
- Determine if additional images are needed for:
  - **Wide-angle (fisheye) cameras**
  - **Omnidirectional cameras**

---

### **Day 3: Curating Chessboard Calibration Data**
- Select high-quality chessboard images with:
  - **Varied angles and distances** (15–20 recommended, 30–40 for better accuracy).
  - **High resolution** to ensure clear detection.
  - **Diverse placements in the frame** (to model distortion).
- Verify that selected images are:
  - **License-compatible** for public use.
  - **Suitable for OpenCV’s calibration pipeline**.

---

### **Day 4: Collecting Data for Other Fiducial Markers**
- Gather data for:
  - **Circle grid patterns (symmetric/asymmetric)**
  - **ArUco and ChArUco boards**
  - **Random dot pattern**
- Generate missing calibration patterns using OpenCV tools:
  - **Chessboards, circles, and ChArUco** (using OpenCV’s pattern generation scripts).
  - **ArUco board generation** (via OpenCV’s `aruco` module or printed designs).
- Acquire images of each pattern from different perspectives and cameras.
- Ensure correct dictionary and board configurations for ChArUco (e.g., **5×7 ChArUco board** as per OpenCV docs).
- Include at least one dataset for **fisheye camera calibration** (captured using a chessboard or ChArUco with a wide FOV lens).
- Identify reliable sources:
  - **OpenCV documentation and tutorials**
  - **Existing ArUco board samples in OpenCV contrib modules**

---

### **Day 5: Evaluating Dataset Quality and Storage**
- Verify dataset quality:
  - Ensure **sharp images** with **clear corners** and **minimal motion blur**.
  - Confirm **full visibility of calibration patterns**.
- Perform a **quick manual calibration**:
  - Test corner detection accuracy (e.g., using `findChessboardCorners`).
  - Validate that images produce reasonable calibration results.
- Assess **dataset size**:
  - Avoid excessive storage use in test suites.
  - Apply strategies like **downsampling** or selecting a **subset of images**.
- Confirm OpenCV’s **test data storage policies**:
  - Test images should be stored in **`opencv_extra/testdata`**.
  - OpenCV tests are **offline** (cloud storage is not used).
  - Large files should be **avoided** or **generated dynamically**.

