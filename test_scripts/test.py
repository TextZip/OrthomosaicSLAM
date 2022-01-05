import cv2
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv2.imread('images/DSC02930.JPG')
# plt.imshow(img_1)
# plt.show()
img_2 = cv2.imread('images/DSC02931.JPG')
# plt.imshow(img_2)
# plt.show()

img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
plt.imshow(img1)
plt.show()
img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
plt.imshow(img2)
plt.show()

sift = cv2.SIFT_create() 
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2, k=2)

good = []
for m in matches:
    if (m[0].distance < 0.5*m[1].distance):
        good.append(m)
matches = np.asarray(good)


if (len(matches[:,0]) >= 4):
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    raise AssertionError('Can’t find enough keypoints.')

dst = cv2.warpPerspective(img_1,H,((img_1.shape[1] + img_2.shape[1]), img_2.shape[0])) #wraped image
dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2 #stitched image
cv2.imwrite('output.jpg',dst)
plt.imshow(dst)
plt.show()