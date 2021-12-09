import cv2
import numpy as np

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img = cv2.imread('images/admin_block.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()

keypoints = surf.detect(img_gray, None)
cv2.imshow("output", cv2.drawKeypoints(img, keypoints, None, (255, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()
