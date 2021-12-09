import cv2
import numpy as np

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img_1 = cv2.imread('images/admin_block.jpg')

orb = cv2.ORB_create(nfeatures=1000)
keypoints_orb, descriptors = orb.detectAndCompute(img_1, None)
cv2.imshow("output", cv2.drawKeypoints(
    img_1, keypoints_orb, None, (255, 0, 255)))
cv2.waitKey(0)
cv2.destroyAllWindows()
