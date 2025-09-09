import cv2
import numpy as np
import matplotlib.pyplot as plt

path = input("Enter image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize=3)
sobel = np.sqrt(sobelx**2 + sobely**2)

prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
prewittx = cv2.filter2D(img, -1, prewitt_x)
prewitty = cv2.filter2D(img, -1, prewitt_y)
prewitt = prewittx + prewitty

plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(sobel, cmap="gray"), plt.title("Sobel")
plt.subplot(1,3,3), plt.imshow(prewitt, cmap="gray"), plt.title("Prewitt")
plt.show()
