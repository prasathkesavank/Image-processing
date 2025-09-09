import cv2
import matplotlib.pyplot as plt

path = input("Enter industrial image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

gauss = cv2.GaussianBlur(img, (5,5), 1)
mask = img - gauss

unsharp = img + mask
high_boost = img + 2*mask  # k=2

plt.subplot(1,3,1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(unsharp, cmap="gray"), plt.title("Unsharp Masking")
plt.subplot(1,3,3), plt.imshow(high_boost, cmap="gray"), plt.title("High-Boost")
plt.show()
