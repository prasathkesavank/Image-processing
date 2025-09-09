import cv2
import numpy as np
import matplotlib.pyplot as plt

path = input("Enter industrial image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), np.uint8)
maxf = cv2.dilate(img, kernel)  # Max filter
minf = cv2.erode(img, kernel)   # Min filter

plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(maxf, cmap="gray"), plt.title("Max Filter")
plt.subplot(1,3,3), plt.imshow(minf, cmap="gray"), plt.title("Min Filter")
plt.show()
