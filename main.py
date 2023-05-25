import sys
import cv2
import numpy as np
import imutils
from imutils import paths
import argparse


class Orthomosaic:
    def __init__(self, debug):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.no_raw_images = []
        self.temp_image = []
        self.final_image = []
        self.debug = debug
        pass

    def load_dataset(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-i", "--images", type=str, required=True,
                             help="path to input directory of images to stitch")
        self.ap.add_argument("-o", "--output", type=str, required=True,
                             help="path to the output image")
        self.args = vars(self.ap.parse_args())

        # grab the paths to the input images and initialize our images list
        if self.debug:
            print("[INFO] Importing Images...")
        self.imagePaths = sorted(list(paths.list_images(self.args["images"])))
        self.images = []
        for imagePath in self.imagePaths:
            self.image_temp = cv2.imread(imagePath)
            scale_percent = 50 # percent of original size
            width = int(self.image_temp.shape[1] * scale_percent / 100)
            height = int(self.image_temp.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            self.image = cv2.resize(self.image_temp, dim)
            # self.image = imutils.resize(self.image_temp, width=500)
            self.images.append(self.image)
        if self.debug:
            print("[INFO] Importing Complete")

        # cv2.imshow("output",self.images[1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def mixer(self):
        self.no_raw_images = len(self.images)
        if self.debug:
            print(f"[INFO] {self.no_raw_images} Images have been loaded")
        for x in range(self.no_raw_images):
            if x == 0:
                self.temp_image = self.sticher(self.images[x],self.images[x+1])
            elif x < self.no_raw_images-1 :
                self.temp_image = self.sticher(self.temp_image,self.images[x+1])
            else:
                self.final_image = self.temp_image                

        # self.final_image = self.sticher(self.images[0], self.images[1])
        cv2.imshow("output", self.final_image)
        cv2.imwrite("output.png", self.final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def sticher(self, image1, image2):
        # image1_grayscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # image2_grayscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        self.image1 = image1
        self.image2 = image2
        orb = cv2.ORB_create(nfeatures=1000)
        print(self.image1.shape)

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(self.image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(self.image2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        all_matches = []
        for m, n in matches:
            all_matches.append(m)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)

        # Set minimum match condition
        MIN_MATCH_COUNT = 0

        if len(good) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Establish a homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            result = self.wrap_images(image2, image1, M)
            # cv2.imwrite('test4.jpg',result)
            # cv2.imshow("output_image",result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return result
        else:
             print("Error")
             pass

    def wrap_images(self, image1, image2, H):
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]
        H = H
        list_of_points_1 = np.float32(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        # When we have established a homography we need to warp perspective
        # Change field of view
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
        list_of_points = np.concatenate(
            (list_of_points_1, list_of_points_2), axis=0)
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]

        H_translation = np.array([[1, 0, translation_dist[0]], [
                                 0, 1, translation_dist[1]], [0, 0, 1]])
        output_img = cv2.warpPerspective(
            image2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        output_img[translation_dist[1]:rows1+translation_dist[1],
                   translation_dist[0]:cols1+translation_dist[0]] = image1
        return output_img

# initialize OpenCV's image stitcher object and then perform the image
# stitching


if __name__ == "__main__":
    tester = Orthomosaic(debug=True)
    tester.load_dataset()
    tester.mixer()
else:
    pass
