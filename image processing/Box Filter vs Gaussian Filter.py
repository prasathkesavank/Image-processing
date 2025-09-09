import cv2
import matplotlib.pyplot as plt

path = input("Enter image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

box = cv2.blur(img, (5,5))
gauss = cv2.GaussianBlur(img, (5,5), 1)

plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(box, cmap="gray"), plt.title("Box Filter")
plt.subplot(1,3,3), plt.imshow(gauss, cmap="gray"), plt.title("Gaussian Filter")
plt.show()
