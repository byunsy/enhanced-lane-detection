"""============================================================================
PART 02. lane_perspective.py
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

"""============================================================================
PROCEDURE:
    transform_lane
PARAMETERS:
    img, a source image
PURPOSE:
    calculates the transformation matrix and warps the region of interest
    within the given image
PRODUCES:
    warped, a warped and cropped image
    pers, a tranformation matrix
    pers_inv, an inverse of a transformation matrix
============================================================================"""
def transform_lane(img):

    # Attain height and width information of image
    img_h, img_w = img.shape[:2]

    # Select source and destination vertices
    # numbers below were used to select the roi region vertices
    src = np.array([[int(0.45 * img_w), int(img_h*0.63)],
                    [int(0.55 * img_w), int(img_h*0.63)],
                    [int(0.88 * img_w), img_h - 50],
                    [int(0.12 * img_w), img_h - 50]], np.float32)

    dst = np.array([[int(0.10 * img_w), 0],
                    [int(0.90 * img_w), 0],
                    [int(0.90 * img_w), img_h - 50],
                    [int(0.10 * img_w), img_h - 50]], np.float32)

    # Attain transformation matrix (as well as its inverse)
    pers = cv2.getPerspectiveTransform(src, dst)
    pers_inv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image using the matrix attained above
    warped = cv2.warpPerspective(img, pers, (img_w, img_h - 50))

    # Display lines to represent selected region of interest
    # cv2.polylines(img, np.int32([src]), True, (255, 255, 0), 3)
    # cv2.polylines(warped, np.int32([dst]), True, (255, 255, 0), 3)

    return warped

"""============================================================================
                                     MAIN
============================================================================"""
def main():

    # Read image
    img = cv2.imread("./images/straight_lines1.jpg")

    # Check for any errors loading images
    if img is None:
        print("Error: Failed to load image.")
        sys.exit()

    warped_lane = transform_lane(img)

    # Show original image
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    # Show warped image
    cv2.namedWindow("warped_lane")
    cv2.imshow("warped_lane", warped_lane)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()