"""============================================================================
PART 07. display_lane.py
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import lane_perspective as lp
import lane_threshold as lt
import calc_radius as cr
import curved_lanes as cl

"""============================================================================
                                     MAIN
============================================================================"""

def main():

    # Create class object of VideoCapture
    cap = cv2.VideoCapture("./video/project_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 25
    dim = (1280, 720)

    out = cv2.VideoWriter("./video/output1.avi", fourcc, fps, dim)

    # Check for any errors opening the video
    if not cap.isOpened():
        print("Error: Failed to open video.")
        sys.exit()

    while True:

        # Keep reading the frames
        ret, frame = cap.read()

        # Exit when video finishes
        if not ret:
            print("Video has ended.")
            sys.exit()

        # Attain height and width information about the image
        img_h_w = frame.shape[:2]

        # Warp the lanes and binarize using the combined threshold
        warped_lane, _, pers_inv = lp.transform_lane(frame)
        warped_lane_bi = lt.combined_threshold(warped_lane)

        # Find curvature information from the warped image
        fit_polynomial = cl.find_curvature(warped_lane_bi)

        # Calculate and display the curved lanes on the warped image
        plot_data, fit_data = cl.find_curved_lanes(warped_lane_bi,
                                                   fit_polynomial)

        # Highlight the calculated lanes in green
        result = cl.highlight_lane(frame, warped_lane_bi, pers_inv, plot_data)

        # Calculate radius of curvature for left and right line (and average)
        curve_data = cr.calc_rad_of_curvature(plot_data)

        # Calculate offset from the center in meters
        offset_m = cr.calc_offset(img_h_w, plot_data, fit_data)

        # Display the curvature and offset information on the result image
        cr.display_info(result, curve_data, offset_m)

        # Show
        cv2.namedWindow("result")
        cv2.imshow("result", result)

        # Write video output
        # out.write(result)

        # Breaking out of the loop
        key = cv2.waitKey(1)

        if key == 27:  # ESC key
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()