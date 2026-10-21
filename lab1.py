#LAB 1
# Display following image operation in MATLAB/Python - i) Histogram  
# image  ii) Low pass filter mage  iii) High pass image. 


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 4))

# Histogram
plt.subplot(1, 3, 1)
plt.hist(gray.ravel(), 256, [0, 256], color='red')
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

# Low-Pass Filter
low = cv2.GaussianBlur(gray, (5, 5), 0)
plt.subplot(1, 3, 2)
plt.imshow(low, cmap='gray')
plt.title("Low-Pass Filter")
plt.axis("off")

# High-Pass Filter
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=np.float32)
high = cv2.filter2D(gray, -1, kernel)
plt.subplot(1, 3, 3)
plt.imshow(high, cmap='gray')
plt.title("High-Pass Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
