# simulate_dataset.py

import cv2
import numpy as np
import os

def simulate_lighting_conditions(base_image_path, output_dir):
    """
    Takes a base image and generates variations simulating different lighting.
    """
    # Load base image
    base_img = cv2.imread("s2.png")
    if base_img is None:
        print(f"Error loading {base_image_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(os.path.join(output_dir, '01_original.jpg'), base_img)

    # --- 1. Simulate Bright and Dim Lighting ---
    bright_img = np.clip(base_img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '02_bright.jpg'), bright_img)

    dim_img = np.clip(base_img.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '03_dim.jpg'), dim_img)
    
    # --- 2. Simulate Uneven Illumination (Vignette) ---
    rows, cols = base_img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette_img = base_img.copy()
    
    for i in range(3): 
        vignette_img[:,:,i] = vignette_img[:,:,i] * mask
    cv2.imwrite(os.path.join(output_dir, '04_vignette_uneven.jpg'), vignette_img)

    # --- 3. Simulate Directional Lighting (Gradient) ---
    gradient_mask = np.linspace(0.7, 1.3, cols) 
    directional_img = base_img.astype(np.float32)
    for i in range(3):
        directional_img[:,:,i] *= gradient_mask
    directional_img = np.clip(directional_img, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '05_directional_left_to_right.jpg'), directional_img)

    # --- 4. Simulate different color temperature (warm/cool) ---
    warm_img = base_img.copy()
    warm_img[:, :, 2] = np.clip(warm_img[:, :, 2] * 1.2, 0, 255)
    warm_img[:, :, 0] = np.clip(warm_img[:, :, 0] * 0.8, 0, 255)
    cv2.imwrite(os.path.join(output_dir, '06_warm_light.jpg'), warm_img)
    
    cool_img = base_img.copy()
    cool_img[:, :, 0] = np.clip(cool_img[:, :, 0] * 1.2, 0, 255)
    cool_img[:, :, 2] = np.clip(cool_img[:, :, 2] * 0.8, 0, 255)
    cv2.imwrite(os.path.join(output_dir, '07_cool_light.jpg'), cool_img)

    print(f"Successfully generated 7 simulated images in '{output_dir}'")


if __name__ == "__main__":
    base_image = 'base_industrial_image.jpg'
    output_folder = 'simulated_dataset'

    try:
        cv2.imread(base_image).shape
    except AttributeError:
        print(f"'{base_image}' not found. Creating a synthetic PCB image.")
        pcb = np.full((400, 600, 3), (20, 80, 20), np.uint8) # Dark green base
        cv2.rectangle(pcb, (50, 50), (200, 150), (180, 180, 180), -1)
        cv2.rectangle(pcb, (250, 100), (550, 120), (200, 150, 50), -1)
        cv2.circle(pcb, (100, 250), 30, (50, 50, 50), -1)
        cv2.circle(pcb, (300, 300), 40, (50, 50, 50), -1)
        cv2.imwrite(base_image, pcb)

    simulate_lighting_conditions(base_image, output_folder)