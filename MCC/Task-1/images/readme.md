<details>
   <summary> chess board generation code </summary>

   ```py

import cv2
import numpy as np
import os

# Define save path
save_path = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/chessboard.png"

# Define pattern size (number of inner corners)
rows, cols = 7, 9  # Change as needed
square_size = 50  # Size in pixels

# Create an empty chessboard
chessboard = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)

# Fill squares
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            cv2.rectangle(
                chessboard, 
                (j * square_size, i * square_size),
                ((j + 1) * square_size, (i + 1) * square_size), 
                (255, 255, 255), 
                -1
            )

# Save the pattern
cv2.imwrite(save_path, chessboard)
print(f"Saved: {save_path}")
```
</details>


<details>
   <summary> symmentric circular grids </summary>

```py

import cv2
import numpy as np

# Define save path
save_path = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/symmetric_circles.png"

# Define grid parameters
rows, cols = 5, 7
dot_radius = 40  # Increased for better clarity
spacing = 150  # More spacing to reduce artifacts
img_size = (cols * spacing, rows * spacing)  # Image size based on grid

# Create blank white image
img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

# Draw high-quality circles with anti-aliasing
for i in range(rows):
    for j in range(cols):
        center = (j * spacing + spacing // 2, i * spacing + spacing // 2)
        cv2.circle(img, center, dot_radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)

# Save high-resolution image
cv2.imwrite(save_path, img)
print(f"Saved: {save_path}")

```

</details>


<details>
   <summary> Random dots </summary>

   ```py

import cv2
import numpy as np

# Define save path
save_path = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/random_dots.png"

# Define image size
height, width = 1000, 1000
num_dots = 200  # Adjust the number of dots
min_spacing = 40  # Minimum spacing between dots

# Create blank white image
img = np.ones((height, width, 3), dtype=np.uint8) * 255

# Generate random dot positions with spacing constraints
dot_positions = []
while len(dot_positions) < num_dots:
    x = np.random.randint(min_spacing, width - min_spacing)
    y = np.random.randint(min_spacing, height - min_spacing)
    radius = np.random.randint(10, 20)  # Random dot size
    
    # Check distance to avoid overlap
    if all(np.linalg.norm(np.array([x, y]) - np.array(pt[:2])) > min_spacing for pt in dot_positions):
        dot_positions.append((x, y, radius))

# Draw high-quality circles
for x, y, radius in dot_positions:
    cv2.circle(img, (x, y), radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)

# Save image
cv2.imwrite(save_path, img)
print(f"Saved: {save_path}")

```
</details>


<details>
   <summary> asymmetric circles grid </summary>

   ```py
iimport cv2
import numpy as np

# Define save path
save_path = "/Users/sankarsanbisoyi/Desktop/Dataset/shyama's dataset/asymmetric_circles.png"

# Define grid parameters (OpenCV-compatible)
rows = 7  # Number of rows
cols = 10  # Number of columns
dot_radius = 30  # Size of each dot
spacing_x = 100  # Horizontal spacing
spacing_y = 100  # Vertical spacing
padding = 50  # Space around the grid to avoid clipping

# Calculate image size with padding
height = (rows - 1) * spacing_y + dot_radius * 2 + 2 * padding
width = (cols - 1) * spacing_x + dot_radius * 2 + 2 * padding

# Create a blank white image
img = np.ones((height, width, 3), dtype=np.uint8) * 255

# Draw high-quality asymmetric circles
for i in range(rows):
    for j in range(cols):
        x_offset = (i % 2) * (spacing_x // 2)  # Shift alternate rows
        center_x = j * spacing_x + x_offset + padding
        center_y = i * spacing_y + padding
        cv2.circle(img, (center_x, center_y), dot_radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)

# Save the high-resolution image
cv2.imwrite(save_path, img)
print(f"Saved: {save_path}")


```

</details>

