"""============================================================================
PART 03. birds_eye_view.py
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

"""============================================================================
                                     MAIN
============================================================================"""
def main():
    
    # Create class object of VideoCapture
    cap = cv2.VideoCapture("./video/test_video3.mp4")

    # Check for any errors opening the camera
    if not cap.isOpened():
        print("Error: Failed to open video.")
        sys.exit()

    while True:

        # Keep reading the frames
        ret, img = cap.read()

        # Exit when video finishes
        if not ret:
            print("Video has ended.")
            sys.exit()

        # # For images
        # img = cv2.imread("./lane_detection/straight_lines1.jpg")

        # # Check for any errors loading images
        # if img is None:
        #     print("Error: Failed to load image.")
        #     sys.exit()

        # Attain height and width information of image
        img_h, img_w = img.shape[:2]
        img_h_w = img.shape[:2]

        src = np.float32([[0, img_h],
                        [img_w, img_h],
                        [img_w, 0],
                        [0, 0]])

        dst = np.float32([[int(0.475 * img_w), img_h],
                        [int(0.525 * img_w), img_h],
                        [img_w, 0],
                        [0, 0]])

        # The transformation matrix
        pers = cv2.getPerspectiveTransform(src, dst)

        # Apply np slicing for ROI crop
        img = img[350:img_h, 0:img_w]

        # Image warping
        warped = cv2.warpPerspective(img, pers, (img_w, img_h))

        cv2.namedWindow("warped")
        cv2.imshow("warped", warped)

        # Breaking out of the loop
        key = cv2.waitKey(10)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()