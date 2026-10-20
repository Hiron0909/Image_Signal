# Lab 3
# Write a MATLAB/Python program to read an image and perform 
# convolution with 3X3 mask. 


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread(r"images.jfif", 0)

# Manually create 3x3 mask
mask = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Perform convolution
output = cv2.filter2D(image, cv2.CV_32F, mask)

# Convert output to visible image
output = cv2.convertScaleAbs(output)

# Display images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Convolution Output")
plt.axis("off")

plt.tight_layout()
plt.show()
