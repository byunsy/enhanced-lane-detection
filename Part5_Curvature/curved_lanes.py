"""============================================================================
PART 05. curved_lanes.py
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import lane_perspective as lp
import lane_threshold as lt

"""============================================================================
PROCEDURE:
    find_curvature
PARAMETERS:
    img, a perspective-transformed binary image
PURPOSE:
    calculate best-fit polynomials for left and right lanes as detected from
    the binary image given
PRODUCES:
    fit_polynomial, a tuple holding polynomials for left and right lanes as
    well as 0 or 1 to indicate whether window center points were used.
    e.g. when center points were used : (ctr_left_fit, ctr_right_fit, 1)
         when nonzero pixels were used: (left_fit, right_fit, 0)
============================================================================"""
def find_curvature(img):

    # Histogram can help us determine the intensity of white pixels(255)
    # - As pixel values are summed up vertically, regions where lanes lie 
    #   will have significantly higher peaks (x-axis:width, y-axis:sum)
    # - Mostly two peaks; one on the left and one the right
    # - Summing up values in the lower half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # Create an output image for result visualization
    # img.shape = (670, 1280); out_img.shape = (670, 1280, 3)
    out_img = np.dstack((img, img, img))*255

    # Calculate the midpoint of width of histogram
    midpoint = np.int(histogram.shape[0] // 2)

    # Calculate the max on the left and right side of the midpoint
    # - this value will be the x value of the peak point
    left_peak_x  = np.argmax(histogram[:midpoint])
    right_peak_x = np.argmax(histogram[midpoint:]) + midpoint

    # - distance between them should be at least 700 and at most 850 pixels
    #   to ensure that they are lanes (if not, set arbitrary x value)
    if ((right_peak_x - left_peak_x) < 700 or 
        (right_peak_x - left_peak_x) > 850):
        left_peak_x  = 275
        right_peak_x = 1100
        print("Lane Distance reconfigured.")

    # Set the number of sliding windows
    window_num = 9

    # Set the window height
    window_height = np.int(img.shape[0] // window_num)

    # Identify x and y coordinates of all nonzero pixels in the image
    nonzero = img.nonzero()             # nonozero = ((array), (array))
    nonzero_x = np.array(nonzero[1])    # x coordinates of nonzero pixels
    nonzero_y = np.array(nonzero[0])    # y coordinates of nonzero pixels

    # Current positions to be updated for each sliding window
    # First begin at the x value of the histogram peak point
    left_x_current  = left_peak_x
    right_x_current = right_peak_x

    # Set the width of the windows +/- margin
    margin = 60

    # Set minimum number of pixels found to recenter window
    min_pix = 40

    # Count the number of times windows have been recentered
    num_win_moved_left = 0
    num_win_moved_right = 0

    # Create empty lists to receive left and right lane pixel indices
    left_lane_index = []
    right_lane_index = []

    # Create empty ndarray to store window center points
    win_ctr_x_left = np.array([])   # x val of center point on left
    win_ctr_x_right = np.array([])  # x val of center point on right
    win_ctr_y = np.array([])        # y val of center point (same for both)

    # For each sliding window
    for window in range(window_num):

        # Identify window boundaries in x and y (for right and left side)
        win_y_low  = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        
        win_x_left_low  = left_x_current - margin
        win_x_left_high = left_x_current + margin
        
        win_x_right_low  = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_left_low, win_y_low),
                               (win_x_left_high, win_y_high),
                               (0,255,0), 2)

        cv2.rectangle(out_img, (win_x_right_low, win_y_low), 
                               (win_x_right_high, win_y_high),
                               (0,255,0), 2) 

        # (win_x_left_low, win_y_low)  
        #            OR
        # (win_x_right_low, win_y_low)
        #                           o --------
        #                           |         |
        #                            -------- o 
        #                                     (win_x_left_high, win_y_high)
        #                                                   OR
        #                                     (win_x_right_high, win_y_high)

        # Identify indices of the nonzero pixels within the current window
        good_left_inds = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) & 
                          (nonzero_x >= win_x_left_low) & 
                          (nonzero_x < win_x_left_high)).nonzero()[0]
       
        good_right_inds = ((nonzero_y >= win_y_low) & 
                           (nonzero_y < win_y_high) & 
                           (nonzero_x >= win_x_right_low) &  
                           (nonzero_x < win_x_right_high)).nonzero()[0]

        # Add these indices to the lists
        left_lane_index.append(good_left_inds)
        right_lane_index.append(good_right_inds)

        # If exceeds min_pix, recenter next window on their mean x position
        # - nonzero_x[good_left_inds] = all x values of nonzero pixels 
        #                               in the current window on the left
        if len(good_left_inds) > min_pix:
            left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            num_win_moved_left += 1

        if len(good_right_inds) > min_pix:        
            right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
            num_win_moved_right += 1
        
        # Append the center points to the existing array
        win_ctr_x_left = np.append( win_ctr_x_left,
                                    (win_x_left_high + win_x_left_low)//2 )  
        
        win_ctr_x_right = np.append( win_ctr_x_right,
                                     (win_x_right_high + win_x_right_low)//2 )

        win_ctr_y = np.append( win_ctr_y,
                               (win_y_high + win_y_low)//2 ) 

    # Concatenate the arrays of indices into one large array
    left_lane_index = np.concatenate(left_lane_index)
    right_lane_index = np.concatenate(right_lane_index)

    # All the xy-coordinates of nonzero pixels within all windows
    # - these points will later be colored red and blue
    left_nz_x = nonzero_x[left_lane_index]
    left_nz_y = nonzero_y[left_lane_index] 

    right_nz_x = nonzero_x[right_lane_index]
    right_nz_y = nonzero_y[right_lane_index]
    
    # =================================================
    # This part is specifically for cases when the number of pixels within 
    # windows are not high enough to accurately produce a best fit.
    # In other words, if the majority of windows do not recenter, which
    # can be determined by num_win_moved_right/left, this suggests that 
    # there are not enough significant pixels around to determine the 
    # general direction of a lane. Hence, I decided that it is safer to
    # rely on the best-fit line based on center points of all windows.

    # Compute a second-order polynomial for best-fit line 
    # through the window center points
    if num_win_moved_left <= 4 or num_win_moved_right <= 4:
        ctr_left_fit = np.polyfit(win_ctr_y, win_ctr_x_left, 2)
        ctr_right_fit = np.polyfit(win_ctr_y, win_ctr_x_right, 2)
        
        # Generate x and y values for plotting
        # - an array = [0, 1, 2, ..., 669]
        plot_y2 = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        # - all x coordinates of the best-fit line calculated above
        left_fit_x2  = (ctr_left_fit[0] * plot_y2**2 + 
                        ctr_left_fit[1] * plot_y2 + 
                        ctr_left_fit[2])

        right_fit_x2 = (ctr_right_fit[0] * plot_y2**2 + 
                        ctr_right_fit[1] * plot_y2 + 
                        ctr_right_fit[2])

        # Visualize all nonzero pixels outside windows
        out_img[nonzero_y, nonzero_x] = [255, 255, 255]  # white : others
        
        # Visualize all nonzero pixels inside windows
        out_img[left_nz_y, left_nz_x] = [255, 0, 0]      # red: left
        out_img[right_nz_y, right_nz_x] = [0, 0, 255]    # blue: right

        plt.imshow(out_img)
        
        # Plot the best-fit line for left and right lane
        plt.plot(left_fit_x2, plot_y2, color='magenta')
        plt.plot(right_fit_x2, plot_y2, color='magenta')

        # Set x and y axis boundaries
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fit_polynomial = (ctr_left_fit, ctr_right_fit, 1)
        print("best-fit performed using center-points.")

        return fit_polynomial, out_img

    # =================================================
    # Otherwise, 
    else:
        # Compute a second-order polynomial for best-fit line 
        # through the nonzero pixels found above 
        left_fit = np.polyfit(left_nz_y, left_nz_x, 2)
        right_fit = np.polyfit(right_nz_y, right_nz_x, 2)

        # Generate x and y values for plotting
        # - an array = [0, 1, 2, ..., 669]
        plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])   

        # - all x coordinates of the best-fit line calculated above
        left_fit_x  = ( left_fit[0] * plot_y**2 + 
                        left_fit[1] * plot_y + 
                        left_fit[2])
        right_fit_x = ( right_fit[0]* plot_y**2 + 
                        right_fit[1]* plot_y + 
                        right_fit[2])

        # Visualize all nonzero pixels outside windows
        out_img[nonzero_y, nonzero_x] = [255, 255, 255]  # white : others

        # Visualize all nonzero pixels inside windows
        out_img[left_nz_y, left_nz_x] = [255, 0, 0]      # red: left
        out_img[right_nz_y, right_nz_x] = [0, 0, 255]    # blue: right

        plt.imshow(out_img)

        # Plot the best-fit line for left and right lane
        plt.plot(left_fit_x, plot_y, color='yellow')
        plt.plot(right_fit_x, plot_y, color='yellow')

        # Set x and y axis boundaries
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fit_polynomial = (left_fit, right_fit, 0)

        return fit_polynomial, out_img

"""============================================================================
PROCEDURE:
    find_curved_lanes
PARAMETERS:
    img, a warped binary image
    polynomial, best-fit polynomial from find_curvature()
PURPOSE:
    uses the best-fit polynomial from find_curvature() to find another 
    best-fit line based on nonzero pixels within margins and generate plot data
    points to draw the curved lanes
PRODUCES:
    - plot_data, a tuple that contains plotting points for y values, left-fit 
      x values, and right-fit x values. (ploty, left_fitx, right_fitx)
    - fit_data, a tuple that contains polynomial information for left and right
      best-fit lines. (left_fit, right_fit)
============================================================================"""
def find_curved_lanes(img, polynomial, out_img):

    # Set the lane margin
    margin = 100

    # Identify x and y coordinates of all nonzero pixels in the image
    nonzero = img.nonzero()             # nonozero = ((array), (array))
    nonzero_x = np.array(nonzero[1])    # x coordinates of nonzero pixels
    nonzero_y = np.array(nonzero[0])    # y coordinates of nonzero pixels

    # Attain best-fit polynomial information from the given parameter
    left_fit, right_fit, used_cp = polynomial

    # Determine whether the nonzero pixels lie within the lane margin
    # - array of booleans
    left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y**2) + 
                                    left_fit[1] * nonzero_y + 
                                    left_fit[2] - margin)) & 
                      (nonzero_x < (left_fit[0] * (nonzero_y**2) + 
                                    left_fit[1] * nonzero_y + 
                                    left_fit[2] + margin)) ) 

    right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y**2) + 
                                     right_fit[1] * nonzero_y + 
                                     right_fit[2] - margin)) & 
                       (nonzero_x < (right_fit[0] * (nonzero_y**2) + 
                                     right_fit[1] * nonzero_y + 
                                     right_fit[2] + margin)) ) 

    # All the xy-coordinates of nonzero pixels within the lane margins
    left_nz_x = nonzero_x[left_lane_inds]
    left_nz_y = nonzero_y[left_lane_inds] 

    right_nz_x = nonzero_x[right_lane_inds]
    right_nz_y = nonzero_y[right_lane_inds]

    # If left/right_fit did not use center points in find_curvature()
    # (ie. used nonzero pixels), then find best-fit based on nonzero pixels
    # within the margins specified above 
    if not used_cp:

        # Compute a second-order polynomial for best-fit line 
        # through the nonzero pixels found within the margins 
        left_fit = np.polyfit(left_nz_y, left_nz_x, 2)
        right_fit = np.polyfit(right_nz_y, right_nz_x, 2)
    
    # Otherwise, use the left/right_fit directly given from the param
    # - this is the best-fit based on window center points

    # Generate x and y values for plotting
    # - an array = [0, 1, 2, ..., 669]
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    # - all x coordinates of the best-fit line calculated above
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Create a blank image
    window_img = np.zeros_like(out_img)

    # Create two lines to bound the highlighted region
    # then change the x and y points into a valid format for fillPoly()

    # np.vstack()    --> [ [x0, x1, x2, ..., xn]    | left_fitx - margin, 
    #                      [y0, y1, y2, ..., yn] ]  | ploty
    # np.transpose() --> [ [x0, y0],                | reverses axis
    #                      [x1, y1],
    #                      [x2, y2],
    #                      ...  ...
    #                      [xn, yn] ]
    # np.flipud()    --> [ [xn, yn],                | reverses order
    #                      ...  ...                 
    #                      [x2, y2],
    #                      [x1, y1],
    #                      [x0, y0] ]

    # left_line_pts after np.hstack()
    # [ [x0, y0],      | left_line_window1       
    #   [x1, y1],               
    #   [x2, y2],           
    #   ...  ...            
    #   [xn, yn],      
    #   [xn, yn],      | left_line_window2  
    #   ...  ...                      
    #   [x2, y2],
    #   [x1, y1],
    #   [x0, y0] ]     | valid format for cv2.fillPoly()

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, 
                                                          ploty] ))])

    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx +
                                                                    margin, 
                                                                    ploty])))])

    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin,
                                                           ploty] ))])

    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + 
                                                                     margin, 
                                                                     ploty])))])

    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    # Blend the highlighted margin window to original image
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    plt.imshow(result)

    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    # Set x and y axis boundaries
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)

    plt.show()

    plot_data = (ploty, left_fitx, right_fitx)
    fit_data  = (left_fit, right_fit)

    return plot_data, fit_data

# Note to self:
# - find_curvature() displays yellow best-fit line based on all nonzero pixels
#   that lie within the windows.
# - on the other hand, find_curved_lane() utilizes best-fit line based on
#   all nonzero pixels that lie within the margin=100 (wider than window)
# - hence, the two best-fit lines are indeed different
# - the difference between these two lines, however, seems indistinguishable 
#   as both of them go in very similar (if not the same) direction.

"""============================================================================
PROCEDURE:
    highlight_lane
PARAMETERS:
    src_img, a source image
    warped_img, a warped binary image
    mat_inv, an inverse transformation matrix calculated from transform_lane()
    plot_data, plotting points calculated from find_curved_lanes()
PURPOSE:
    to draw and display the detected lanes on the actual source image
PRODUCES:
    result, an image with detected lanes highlighted in green
============================================================================"""
def highlight_lane(src_img, warped_img, mat_inv, plot_data):

    # Extract plotting data calculcated from find_curved_lanes()
    ploty, left_fitx, right_fitx = plot_data

    # Create an color image to show visualization
    warp_blank = np.zeros_like(warped_img).astype(np.uint8)
    warp_color = np.dstack((warp_blank, warp_blank, warp_blank))

    # Create two lines to bound the highlighted region
    # then change the x and y points into a valid format for fillPoly()
    # - same method as explained above in find_curved_lanes()
    left_line = np.array([np.transpose(np.vstack([left_fitx, 
                                                  ploty]))])

    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                                                             ploty])))])

    lane_area = np.hstack((left_line, right_line))

    # Highlight the lane onto the blank warped image
    cv2.fillPoly(warp_color, np.int_([lane_area]), (0, 255, 0))

    # Unwarp the highlighted lane image to original image space with the
    # given inverse transformation matrix 
    new_warp = cv2.warpPerspective(warp_color, mat_inv, (src_img.shape[1], 
                                                         src_img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(src_img, 1, new_warp, 0.3, 0)

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return result

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

    # Warp the lanes and binarize using the combined threshold created before
    warped_lane, _, pers_inv = lp.transform_lane(img)
    warped_lane_bi = lt.combined_threshold(warped_lane)

    # Show warped image with gradient thresholds
    cv2.namedWindow("warped_lane_bi")
    cv2.imshow("warped_lane_bi", warped_lane_bi)

    # Find curvature information from the warped image
    fit_polynomial, out_img = find_curvature(warped_lane_bi)

    # Calculate and display the curved lanes on the warped image
    plot_data, _ = find_curved_lanes(warped_lane_bi, fit_polynomial, out_img)
    
    highlight_lane(img, warped_lane_bi, pers_inv, plot_data)


if __name__ == '__main__':
    main()