import cv2
import numpy as np
import matplotlib.pyplot as plt

path = input("Enter document image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

lap = cv2.Laplacian(img, cv2.CV_64F)
enhanced = cv2.convertScaleAbs(img + lap)

plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(lap, cmap="gray"), plt.title("Laplacian")
plt.subplot(1,3,3), plt.imshow(enhanced, cmap="gray"), plt.title("Enhanced")
plt.show()
