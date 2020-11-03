"""============================================================================
PART 04. lane_threshold.py
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

"""============================================================================
PROCEDURE:
    abs_sobel_thresh
PARAMETERS:
    img, a source image
    kernel, (kernel x kernel)-sized kernel used to calculate the derivatives
    thresh_vals, (x_thresh_min, x_thresh_max, y_thresh_min, y_thresh_max)
PURPOSE:
    performs sobel filter in both x and/or y directions using given kernel
    and threshold values parameters
PRODUCES:
    x_binary, a binary image of sobel filter applied in x-direction
    y_binary, a binary image of sobel filter applied in y-direction
REFERENCES:
    about the need for absolute sobel: https://bit.ly/3oNAjU1
============================================================================"""
def sobel_thresh(img, kernel=3, thresh_vals=(20, 110, 20, 110)):

    # Convert source image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take the derivative in both x and y using Sobel method
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    abs_sobel_x = np.absolute(sobel_x)      # absolute values

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    abs_sobel_y = np.absolute(sobel_y)      # absolute values

    # Scale all values to (0 - 255) and convert them to uint8 type
    scaled_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    scaled_sobel_y = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))

    # Create a blank black image with same dimensions
    x_img_h_w = scaled_sobel_x.shape[:2]
    x_binary = np.zeros(x_img_h_w, np.uint8)

    y_img_h_w = scaled_sobel_y.shape[:2]
    y_binary = np.zeros(y_img_h_w, np.uint8)

    # Let values in scaled_sobel be 'val'.
    # if thresh_min <= val <= thresh_max, then set val to 255 (white)
    x_binary[(scaled_sobel_x >= thresh_vals[0]) &
             (scaled_sobel_x <= thresh_vals[1])] = 255

    y_binary[(scaled_sobel_y >= thresh_vals[2]) &
             (scaled_sobel_y <= thresh_vals[3])] = 255

    return x_binary, y_binary

"""============================================================================
PROCEDURE:
    mag_thresh
PARAMETERS:
    img, a source image
    kernel, (kernel x kernel)-sized kernel used to calculate the derivatives
    thresh_min, minimum value to be 255(white) in a binary image
    thresh_max, maximum value to be 255(white) in a binary image
PURPOSE:
    calculates the magnitude of gradient and ultimately shows pixels with
    magnitude values that satisfy the threshold
PRODUCES:
    binary, a binary image based on gradient magnitude values
REFERENCE:
    about gradient magnitude: https://bit.ly/30McyRV
============================================================================"""
def mag_thresh(img, kernel=3, thresh_min=0, thresh_max=255):

    # Convert source image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the derivative in both x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    # Calculate the gradient magnitude
    mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Scale all values to (0 - 255) and convert them to uint8 type
    scaled_mag = np.uint8(255 * mag / np.max(mag))

    # Create a blank black image with same dimensions
    img_h_w = scaled_mag.shape[:2]
    binary = np.zeros(img_h_w, np.uint8)

    # Let values in scaled_mag be 'val'.
    # if thresh_min <= val <= thresh_max, then set val to 255 (white)
    binary[(scaled_mag >= thresh_min) & (scaled_mag <= thresh_max)] = 255

    return binary

"""============================================================================
PROCEDURE:
    dir_thresh
PARAMETERS:
    img, a source image
    kernel, (kernel x kernel)-sized kernel used to calculate the derivatives
    thresh_min, minimum value to be 255(white) in a binary image
    thresh_max, maximum value to be 255(white) in a binary image
PURPOSE:
    calculates the direction of gradient and ultimately shows pixels with
    directional values that satisfy the threshold
PRODUCES:
    binary, a binary image based on gradient direction values
REFERENCE:
    about gradient direction: https://bit.ly/30McyRV
============================================================================"""
def dir_thresh(img, kernel=3, thresh_min=0, thresh_max=(np.pi/2)):

    # Convert source image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the derivative in both x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    abs_sobel_x = np.absolute(sobel_x)      # absolute values

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    abs_sobel_y = np.absolute(sobel_y)       # absolute values

    # Take the absolute value of the gradient direction
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)

    # Create a blank black image with same dimensions
    img_h_w = direction.shape[:2]
    binary = np.zeros(img_h_w, np.uint8)

    # Let values in scaled_mag be 'val'.
    # if thresh_min <= val <= thresh_max, then set val to 255 (white)
    binary[(direction >= thresh_min) & (direction <= thresh_max)] = 255

    return binary

"""============================================================================
PROCEDURE:
    hls_thresh
PARAMETERS:
    img, a source image
    thresh_vals, (h_thresh_min, h_thresh_max, l_thresh_min, l_thresh_max, 
                  s_thresh_min, s_thresh_max)
PURPOSE:
    converts the colorspace to HLS and ultimately shows pixels with
    color values that satisfy the threshold for each channel
PRODUCES:
    h_binary, a binary image based on h channel values
    l_binary, a binary image based on l channel values
    s_binary, a binary image based on s channel values
============================================================================"""
def hls_thresh(img, thresh_vals=(10, 50, 160, 255, 180, 255)):

    # Convert source image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Attain each of the HLS channels separately
    h_ch = hls[:, :, 0]
    l_ch = hls[:, :, 1]
    s_ch = hls[:, :, 2]

    # H channel
    h_img_h_w = h_ch.shape[:2]
    h_binary = np.zeros(h_img_h_w, np.uint8)
    h_binary[(h_ch > thresh_vals[0]) & (h_ch <= thresh_vals[1])] = 255

    # L channel
    l_img_h_w = l_ch.shape[:2]
    l_binary = np.zeros(l_img_h_w, np.uint8)
    l_binary[(l_ch > thresh_vals[2]) & (l_ch <= thresh_vals[3])] = 255

    # S channel
    s_img_h_w = s_ch.shape[:2]
    s_binary = np.zeros(s_img_h_w, np.uint8)
    s_binary[(s_ch > thresh_vals[4]) & (s_ch <= thresh_vals[5])] = 255

    return h_binary, l_binary, s_binary

"""============================================================================
PROCEDURE:
    luv_thresh
PARAMETERS:
    img, a source image
    thresh_vals, (l_thresh_min, l_thresh_max, u_thresh_min, u_thresh_max, 
                  v_thresh_min, v_thresh_max)
PURPOSE:
    converts the colorspace to LUV and ultimately shows pixels with
    color values that satisfy the threshold for each channel
PRODUCES:
    l_binary, a binary image based on l channel values
    u_binary, a binary image based on u channel values
    v_binary, a binary image based on v channel values
============================================================================"""
def luv_thresh(img, thresh_vals=(200, 255, 100, 255, 150, 255)):

    # Convert source image to LUV color space
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    l_ch = luv[:, :, 0]
    u_ch = luv[:, :, 1]
    v_ch = luv[:, :, 2]

    # L channel
    l_img_h_w = l_ch.shape[:2]
    l_binary = np.zeros(l_img_h_w, np.uint8)
    l_binary[(l_ch > thresh_vals[0]) & (l_ch <= thresh_vals[1])] = 255

    # U channel
    u_img_h_w = u_ch.shape[:2]
    u_binary = np.zeros(u_img_h_w, np.uint8)
    u_binary[(u_ch > thresh_vals[2]) & (u_ch <= thresh_vals[3])] = 255

    # V channel
    v_img_h_w = v_ch.shape[:2]
    v_binary = np.zeros(v_img_h_w, np.uint8)
    v_binary[(v_ch > thresh_vals[4]) & (v_ch <= thresh_vals[5])] = 255

    return l_binary, u_binary, v_binary

"""============================================================================
PROCEDURE:
    lab_thresh
PARAMETERS:
    img, a source image
    thresh_vals, (l_thresh_min, l_thresh_max, a_thresh_min, a_thresh_max, 
                  b_thresh_min, b_thresh_max)
PURPOSE:
    converts the colorspace to LAB and ultimately shows pixels with
    color values that satisfy the threshold for each channel
PRODUCES:
    l_binary, a binary image based on l channel values
    a_binary, a binary image based on a channel values
    b_binary, a binary image based on b channel values
============================================================================"""
def lab_thresh(img, thresh_vals=(200, 255, 125, 135, 150, 255)):

    # Convert source image to LUV color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    # L channel
    l_img_h_w = l_ch.shape[:2]
    l_binary = np.zeros(l_img_h_w, np.uint8)
    l_binary[(l_ch > thresh_vals[0]) & (l_ch <= thresh_vals[1])] = 255

    # U channel
    a_img_h_w = a_ch.shape[:2]
    a_binary = np.zeros(a_img_h_w, np.uint8)
    a_binary[(a_ch > thresh_vals[2]) & (a_ch <= thresh_vals[3])] = 255

    # V channel
    b_img_h_w = b_ch.shape[:2]
    b_binary = np.zeros(b_img_h_w, np.uint8)
    b_binary[(b_ch > thresh_vals[4]) & (b_ch <= thresh_vals[5])] = 255

    return l_binary, a_binary, b_binary

"""============================================================================
PROCEDURE:
    combined_threshold
PARAMETERS:
    img, a source image
PURPOSE:
    incorporates a number of different threshold functions and combines to 
    a single new threshold function
PRODUCES:
    combined_gradient, a binary image based on a combination of different 
    thresholds
============================================================================"""
def combined_threshold(img):

    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    grad_x, _ = sobel_thresh(img)

    mag_bi = mag_thresh(img, thresh_min=100, thresh_max=200)

    dir_bi = dir_thresh(img, kernel=15, thresh_min=0.2, thresh_max=0.4)

    # HLS Color Threshold
    _, _, s_bi = hls_thresh(img, thresh_vals=(0, 0, 0, 0, 180, 255))

    # LUV Color Threshold
    cieL_bi, _, _ = luv_thresh(img, thresh_vals=(180, 255, 0, 0, 0, 0))

    img_h_w = img.shape[:2]
    combined_gradient = np.zeros(img_h_w, np.uint8)

    combined_gradient[((grad_x == 255) |
                       ((mag_bi == 255) & (dir_bi == 255)) |
                       ((s_bi == 255) & (cieL_bi == 255)))] = 255

    return combined_gradient