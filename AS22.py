# calibrate_camera.py

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Number of inner corners on your checkerboard
CHECKERBOARD = (6, 9) 
SQUARE_SIZE = 25 # mm

def calibrate_and_undistort():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # 2. Find Checkerboard Corners in Images 
    images = glob.glob("download1.jpg")
    if not images:
        print("Error: No calibration images found in 'calibration_images/' folder.")
        print("Please capture checkerboard images and place them there.")
        return

    print("Finding corners in calibration images...")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[1], CHECKERBOARD[0]), None)

        # If found, add object points, refine and add image points
        if ret == True:
            objpoints.append(objp)
            # Refine corner locations for better accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    
    if not objpoints:
        print("Could not find checkerboard in any of the images. Check CHECKERBOARD dimensions.")
        return

    print(f"Found corners in {len(objpoints)} images.")
    
    # 3. Perform Camera Calibration
    print("Calibrating camera...")
    # This function returns the camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Calibration successful!")
    print("Camera Matrix (Intrinsic Parameters):\n", mtx)
    print("\nDistortion Coefficients:\n", dist)

    # 4. Undistort a Test Image 
    test_img_path = "download1.jpg"
    try:
        img_to_undistort = cv2.imread(test_img_path)
        h, w = img_to_undistort.shape[:2]
    except:
        print(f"\n'{test_img_path}' not found. Creating a synthetic distorted grid.")
        img_to_undistort = np.full((480, 640, 3), 255, np.uint8)
        for i in range(0, 640, 40): cv2.line(img_to_undistort, (i, 0), (i, 480), (0,0,0), 1)
        for i in range(0, 480, 40): cv2.line(img_to_undistort, (0, i), (640, i), (0,0,0), 1)
        h, w = img_to_undistort.shape[:2]
        # Artificially distort this grid for demonstration
        img_to_undistort = cv2.remap(img_to_undistort, *cv2.initUndistortRectifyMap(mtx, -dist*2, None, mtx, (w,h), 5), cv2.INTER_LINEAR)


    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    undistorted_img = cv2.undistort(img_to_undistort, mtx, dist, None, newcameramtx)
    
    x, y, w, h = roi
    cropped_undistorted = undistorted_img[y:y+h, x:x+w]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(img_to_undistort, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Distorted Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(cropped_undistorted, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Corrected (Undistorted) Image')
    axes[1].axis('off')
    
    plt.suptitle("Demonstration of Lens Distortion Correction", fontsize=16)
    plt.show()


if __name__ == '__main__':
    calibrate_and_undistort()