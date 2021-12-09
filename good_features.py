import cv2
import numpy as np

img_1 = cv2.imread('images/admin_block.jpg')
gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 1)

corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img_1, (x, y), 3, (255, 0, 255), -1)

cv2.imshow("output", img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
