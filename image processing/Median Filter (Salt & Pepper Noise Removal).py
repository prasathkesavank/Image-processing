import cv2
import matplotlib.pyplot as plt

path = input("Enter noisy PCB image path: ").strip('"')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

median = cv2.medianBlur(img, 5)

plt.subplot(1,2,1), plt.imshow(img, cmap="gray"), plt.title("Noisy")
plt.subplot(1,2,2), plt.imshow(median, cmap="gray"), plt.title("Median Filtered")
plt.show()
