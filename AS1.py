# virtual_lab_demo.py

# Required libraries:
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_demonstrations(image_path):
    # Load images
    bgr_image = cv2.imread("apple.jpg")
    if bgr_image is None:
        print(f"Error loading {image_path}")
        return
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # --- 1. Image Negative ---
    negative_img = 255 - rgb_image

    # --- 2. Histogram Equalization (on grayscale) ---
    equalized_img = cv2.equalizeHist(gray_image)

    # --- 3. Binary Thresholding (on grayscale) ---
    _, threshold_img = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # --- 4. Edge Detection (Sobel) ---
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 5. Filtering (Gaussian Blur) ---
    blurred_img = cv2.GaussianBlur(rgb_image, (15, 15), 0)

    # --- Plotting ---
    titles = [
        'Original RGB', 'Image Negative',
        'Original Gray', 'Histogram Equalized', 'Binary Threshold',
        'Sobel Edge Detection', 'Gaussian Blur'
    ]
    images = [
        rgb_image, negative_img, 
        gray_image, equalized_img, threshold_img, 
        sobel_combined, blurred_img
    ]
    cmaps = ['viridis', 'viridis', 'gray', 'gray', 'gray', 'gray', 'viridis']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    fig.suptitle('Common Image Processing Demonstrations', fontsize=20)
    
    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap=cmap)
        else:
            axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    image_path = 'your_image.jpg'
    try:
        cv2.imread(image_path).shape
    except AttributeError:
        print(f"'{image_path}' not found. Creating a synthetic image.")
        img = np.zeros((300, 300, 3), np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 60, (128, 128, 128), -1)
        cv2.imwrite('your_image.jpg', img)

    run_demonstrations(image_path)