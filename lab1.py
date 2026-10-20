#LAB 1
# Display following image operation in MATLAB/Python - i) Histogram  
# image  ii) Low pass filter mage  iii) High pass image. 


import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("images.jfif")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Histogram
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(gray.ravel(), bins=256, range=[0, 256], color='red')
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

# 2. Low-Pass Filter
low_pass = cv2.GaussianBlur(gray, (5, 5), 0)

plt.subplot(1, 3, 2)
plt.imshow(low_pass, cmap='gray')
plt.title("Low-Pass Filter")
plt.axis("off")

# 3. High-Pass Filter
kernel = [[-1, -1, -1],
          [-1,  8, -1],
          [-1, -1, -1]]

high_pass = cv2.filter2D(gray, -1, 
                         __import__('numpy').array(kernel, dtype='float32'))

plt.subplot(1, 3, 3)
plt.imshow(high_pass, cmap='gray')
plt.title("High-Pass Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
