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