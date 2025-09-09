import cv2
import matplotlib.pyplot as plt

path = input("Enter image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

lap = cv2.Laplacian(img, cv2.CV_64F)
log = cv2.GaussianBlur(img, (5,5), 0)
log = cv2.Laplacian(log, cv2.CV_64F)

plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(lap, cmap="gray"), plt.title("Laplacian")
plt.subplot(1,3,3), plt.imshow(log, cmap="gray"), plt.title("LoG")
plt.show()
