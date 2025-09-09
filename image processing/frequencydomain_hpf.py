import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import data, color

def frequency_domain_hpf(image, cutoff_radius):
    """
    Apply a high-pass filter in the frequency domain to an image.

    Parameters:
    - image: 2D numpy array (grayscale image)
    - cutoff_radius: radius of the low-frequency cutoff (in pixels)

    Returns:
    - filtered_image: high-pass filtered image (real part)
    """
    # Compute the 2D Fourier transform of the image
    f = fft2(image)
    fshift = fftshift(f)  # Shift zero frequency to center

    # Create a high-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with zeros in the low-frequency region (center circle)
    mask = np.ones((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (y - crow)**2 + (x - ccol)**2 <= cutoff_radius**2
    mask[mask_area] = 0  # Block low frequencies

    # Apply mask and inverse FFT
    fshift_filtered = fshift * mask
    f_ishift = ifftshift(fshift_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.real(img_back)

    return img_back

# Example usage
image = color.rgb2gray(data.astronaut())  # Load example image and convert to grayscale
cutoff = 30  # Adjust cutoff frequency radius

filtered_img = frequency_domain_hpf(image, cutoff)

# Plot original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('High-Pass Filtered Image (Frequency Domain)')
plt.imshow(filtered_img, cmap='gray')
plt.axis('off')

plt.show()