# Lab 10
# Perform image enhancement, smoothing and sharpening, in spatial domain 
# using different spatial filters and compare the performances  

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("images.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Image Enhancement
enhanced = cv2.equalizeHist(gray)

# 2. Image Smoothing
smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Image Sharpening
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpened = cv2.convertScaleAbs(gray + laplacian)

# Display results
images = [gray, enhanced, smoothed, sharpened]
titles = ["Original Image", "Enhanced Image",
          "Smoothed Image", "Sharpened Image"]

plt.figure(figsize=(10, 7))

plt.figure(figsize=(10, 7))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Enhanced Image
plt.subplot(2, 2, 2)
plt.imshow(enhanced, cmap="gray")
plt.title("Enhanced Image")
plt.axis("off")

# Smoothed Image
plt.subplot(2, 2, 3)
plt.imshow(smoothed, cmap="gray")
plt.title("Smoothed Image")
plt.axis("off")

# Sharpened Image
plt.subplot(2, 2, 4)
plt.imshow(sharpened, cmap="gray")
plt.title("Sharpened Image")
plt.axis("off")



plt.tight_layout()
plt.show()
