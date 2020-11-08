"""============================================================================
PART 01. cam_calibration.py
============================================================================"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
from pathlib import Path

"""============================================================================
PROCEDURE:
    get_obj_img_pts
PARAMETERS:
    None
PURPOSE:
    attains object and image points using chessboard calibration corners
PRODUCES:
    obj_points, a list for 3D points
    img_points, a list for 2D points
    criteria, the subpixel termination criteria
============================================================================"""
def get_obj_img_pts():

    # Define the dimensions of chessboard (6 x 9)
    board_dim = (6, 9)

    # Define subpixel termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define chessboard flags
    board_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                   cv2.CALIB_CB_FAST_CHECK +
                   cv2.CALIB_CB_NORMALIZE_IMAGE)

    # Object points: from (0,0,0) to (8,5,0)
    obj_p = np.zeros((board_dim[0] * board_dim[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1, 2)

    # Create lists to store 3D points and 2D points for each chessboard
    obj_points = []     # For 3D points
    img_points = []     # For 2D points

    # Create a list for storing calibration images
    calib_images = glob.glob("./distorted/calibration*.jpg")

    # Go through each calibration image files calibrate camera
    for img_file in calib_images:

        # Read each image files
        img = cv2.imread(img_file)

        # Check for any errors loading images
        if img is None:
            print("Error: Failed to load image.")
            sys.exit()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Attain the corners on chessboard using findChessboardCorners()
        ret, corners = cv2.findChessboardCorners(gray, board_dim, board_flags)

        # If desired number of corners are found on a given image
        if ret == True:

            # add object points to the list
            obj_points.append(obj_p)

            # refine and add image points to the list
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            img_points.append(corners2)
   
    return obj_points, img_points

"""============================================================================
PROCEDURE:
    undistort_img
PARAMETERS:
    img, a source image
    obj_points, an array of 3D points in real world space
    img_points, an array of 2D points in real world space
PURPOSE:
    takes a distorted image and undistorts using the parameters given
PRODUCES:
    undistorted_cropped, an undistorted and ROI-cropped image
============================================================================"""
def undistort_img(img, obj_points, img_points):

    # Get grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calibrate camerausing calibrateCamera()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                       img_points,
                                                       gray.shape[::-1],
                                                       None, None)

    # Attain height and width information
    img_h, img_w = img.shape[:2]

    # Refine the camera-matrix based on a free scaling parameter
    # alpha = 0 --> returns undistorted image with minimum unwanted pixels.
    #               *may even remove some pixels at image corners.
    # alpha = 1 --> all pixels are retained with some extra black images.
    #               *creates 'black hills' which can be cropped using roi.
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                        dist,
                                                        (img_w, img_h),
                                                        1,
                                                        (img_w, img_h))

    # Undistort the image using the variables attained above
    undistorted = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # Attain x, y coordinates for left top corner of the ROI as well as
    # the width and height of the ROI
    roi_x, roi_y, roi_w, roi_h = roi

    # Crop the ROI in the undistorted image
    undistorted_cropped = undistorted[roi_y: roi_y + roi_h,
                                      roi_x: roi_x + roi_w]

    return undistorted_cropped


"""============================================================================
                                     MAIN
============================================================================"""
def main():

    # Create a list for storing calibration images
    calib_images = glob.glob("./distorted/calibration*.jpg")

    # Attain object and image points
    obj_points, img_points = get_obj_img_pts()

    for img_file in calib_images:

        # Read image files
        img = cv2.imread(img_file)

        # Check for any errors loading images
        if img is None:
            print("Error: Failed to load image.")
            sys.exit()

        # Generate undistorted image
        undistorted_cropped = undistort_img(img, obj_points, img_points)

        # Save the undistorted image in given directory
        save_name = (Path(img_file).stem)[11:]
        cv2.imwrite("./undistorted/undistorted{}.jpg"
                    .format(save_name), undistorted_cropped)


if __name__ == '__main__':
    main()