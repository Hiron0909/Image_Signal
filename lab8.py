# lab 8
# Write a MATLAB/Python program to read coins.png, leveling all coins and display area of all coins. 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("coins.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Convert to binary image
_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
coins = []

for c in contours:
    area = cv2.contourArea(c)
    if area > 5000:
        (x, y), r = cv2.minEnclosingCircle(c)
        coins.append((int(x), int(y), int(r), area))

# Sort coins
coins.sort(key=lambda c: (c[1], c[0]))

print("Number of coins:", len(coins))

for i, (x, y, r, area) in enumerate(coins, 1):
    cv2.circle(result, (x, y), r, (0, 255, 0), 3)
    cv2.circle(result, (x, y), 5, (255, 0, 0), -1)
    cv2.putText(result, f"Coin {i}: {area:.0f} px",
                (x - 70, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)
    print(f"Coin {i}: Area = {area:.2f} pixels")

# Display images
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Coins (Total: {len(coins)})")
plt.axis("off")

plt.tight_layout()
plt.show()
