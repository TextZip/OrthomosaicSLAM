import cv2
import numpy as np

img = cv2.imread('images/admin_block.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)


sift = cv2.SIFT_create()

keypoints = sift.detect(img_gray, None)
cv2.imshow("output", cv2.drawKeypoints(img, keypoints, None, (255, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()
