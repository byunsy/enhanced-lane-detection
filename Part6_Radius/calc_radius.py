"""============================================================================
PART 06. calc_radius.py
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import lane_perspective as lp
import lane_threshold as lt
import curved_lanes as cl

"""============================================================================
PROCEDURE:
    calc_rad_of_curvature
PARAMETERS:
    plot_data, a tuple that contains plotting points for y values, left-fit 
    x values, and right-fit x values
PURPOSE:
    calculates the radius of curvature to estimate the curvature of the lanes
PRODUCES:
    curve_data, a tuple containing radius of curvature for left and right
    linse as well as the average of the two
References:
    about radius of curvature: https://www.math24.net/curvature-radius/ 
============================================================================"""
def calc_rad_of_curvature(plot_data):

    # Unpack the given tuple parameter
    ploty, left_fitx, right_fitx = plot_data

    # Define conversions in x and y from pixels space to meters
    # - the choices for these numbers are explained below in the references
    ym_per_pix = 36.58 / 670.0              # meters per pixel in y dimension
    xm_per_pix = 3.70 / 800.0               # meters per pixel in x dimension

    # Calculate best-fit based on the new meters per pixels
    left_fit_roc = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_roc = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Evaluation point
    y_eval = np.max(ploty)   # bottom of the image

    # Calculate radius of curvatures for left and right curves
    left_curve_rad = (((1 + (2 * left_fit_roc[0] * y_eval * ym_per_pix +
                             left_fit_roc[1])**2)**1.5) /
                      np.absolute(2 * left_fit_roc[0]))

    right_curve_rad = (((1 + (2 * right_fit_roc[0] * y_eval * ym_per_pix +
                              right_fit_roc[1])**2)**1.5) /
                       np.absolute(2 * right_fit_roc[0]))

    # Average curvature of the lane
    average_curverad = (left_curve_rad + right_curve_rad) / 2

    curve_data = (left_curve_rad, right_curve_rad, average_curverad)

    return curve_data

"""============================================================================
PROCEDURE:
    calc_offset
PARAMETERS:
    - img_h_w, a tuple containing height and width information
    - plot_data, a tuple that contains plotting points for y values, left-fit 
      x values, and right-fit x values
    - fit_data,a tuple that contains polynomial information for left and right
      best-fit lines
PURPOSE:
    to calculate how far the center of a car is from the center of a detected
    lane
PRODUCES:
    offset_m, the calculated offset in meters
============================================================================"""
def calc_offset(img_h_w, plot_data, fit_data):

    # Define conversions in x from pixels space to meters
    xm_per_pix = 3.70 / 700    # meters per pixel in x dimension

    # Unpack parameters
    img_h, img_w = img_h_w           # image height and width
    left_fit, right_fit = fit_data   # left and right fit data

    # Find the x coord of start point of a lane (beginning of best-fit line)
    # - the x val at very bottom of the best-fit line
    left_fit_xbottom = (left_fit[0] * img_h**2 +
                        left_fit[1] * img_h +
                        left_fit[2])

    right_fit_xbottom = (right_fit[0] * img_h**2 +
                         right_fit[1] * img_h +
                         right_fit[2])

    # Calculate the center of the two bottom points
    lane_center = (left_fit_xbottom + right_fit_xbottom) / 2.

    # Calculate how far off lane_center is from the center of image
    offset_pix = (img_w / 2) - lane_center

    # Convert units (from pixels to meters)
    offset_m = offset_pix * xm_per_pix

    return offset_m

"""============================================================================
PROCEDURE:
    display_info
PARAMETERS:
    - img, a source image
    - curve_data, a tuple containing radius of curvature for left and right
      linse as well as the average of the two -- from calc_rad_of_curvature()
    - offset, the calculated offset in meters -- from calc_offset()
PURPOSE:
    displays all the calculated information on the window
PRODUCES:
    none - a void function    
============================================================================"""
def display_info(img, curve_data, offset):

    # Unpack the parameters:
    #   left_curve_rad, right_curve_rad, and average_curve_rad
    lcr, rcr, acr = curve_data

    # Create string variables to display
    str_lcr = "Left Radius of Curvature : %.2f m" % lcr
    str_rcr = "Right Radius of Curvature : %.2f m" % rcr
    str_acr = "Average Radius of Curvature : %.2f m" % acr
    str_offset = "Center Offset: %.2f m" % offset

    # Put the string variables on the screen
    cv2.putText(img, str_lcr, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)
    cv2.putText(img, str_rcr, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)
    cv2.putText(img, str_acr, (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)
    cv2.putText(img, str_offset, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)

    # Draw a red line at the center of screen (center of car)
    img_h, img_w = img.shape[:2]
    cv2.line(img, (img_w//2, img_h - 65),
             (img_w//2, img_h - 55), (0, 0, 255), thickness=2)

    # Ratio calculated previously 
    xm_per_pix = 3.70 / 700
    offset_pix = offset / xm_per_pix

    # Draw a red line from center of a car to center of a lane
    cv2.line(img, (img_w//2, img_h - 60), (img_w//2 - int(offset_pix), img_h - 60),
             (0, 0, 255), thickness=2)

    # Draw a green line at the center of a lane
    cv2.line(img, (img_w//2 - int(offset_pix), img_h - 70), (img_w//2 - int(offset_pix), img_h - 50),
             (0, 255, 0), thickness=4)

    # Show
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""============================================================================
                                     MAIN
============================================================================"""
def main():

    # Read image
    img = cv2.imread("./images/test2.jpg")

    # Check for any errors loading images
    if img is None:
        print("Error: Failed to load image.")
        sys.exit()

    # Attain height and width information about the image
    img_h_w = img.shape[:2]

    # Warp the lanes and binarize using the combined threshold created before
    warped_lane, _, pers_inv = lp.transform_lane(img)
    warped_lane_bi = lt.combined_threshold(warped_lane)

    # Find curvature information from the warped image
    fit_polynomial = cl.find_curvature(warped_lane_bi)

    # Calculate and display the curved lanes on the warped image
    plot_data, fit_data = cl.find_curved_lanes( warped_lane_bi,
                                                fit_polynomial)

    # calculate radius of curvature for left and right line (and average)
    curve_data = calc_rad_of_curvature(plot_data)

    # calculate offset from the center in meters
    offset_m = calc_offset(img_h_w, plot_data, fit_data)

    # Highlight the calculated lanes in green
    result = cl.highlight_lane(img, warped_lane_bi, pers_inv, plot_data)

    # Display the curvature and offset information on the result image
    display_info(result, curve_data, offset_m)


if __name__ == '__main__':
    main()



""" 
REFERENCES:
Choices for these specific numbers:
 - ym_per_pix = 36.58 / 670.0            
 - xm_per_pix = 3.70 / 800.0 

Based on https://bit.ly/3oRNEKO (Federal Highway Administration), the lane
width for U.S. Highways are set at 12ft which is about 3.7m. The warped
binary images, on the other hand, have shown that lanes are about 800 pixels 
apart. Hence, I used the ratio of 3.70 / 800.0 

Based on a diagram from https://bit.ly/3jWEk52 (Texas Dpt. of Transportation), 
the white lane markings are 30ft apart and the markings themselves are 10ft.
Looking at the the warped binary images, the region of interest I set
corresponds to about 120ft which is about 36.58m. The height of the binary 
image is 670 pixels. Hence, I used the ratio of 36.58 / 670.0        
"""