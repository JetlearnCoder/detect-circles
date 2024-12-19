import cv2
import numpy as np


"""image1 = cv2.imread("pikachu.png",1)"""
"""image1 = cv2.imread("sonic.png",1)"""
image1 = cv2.imread("zcircless.png",1)

paramas = cv2.SimpleBlobDetector_Params()

paramas.filterByArea = True
paramas.minArea = 3

paramas.filterByConvexity = True
paramas.minConvexity = 0.1

paramas.filterByInertia = True
paramas.minInertiaRatio = 0.2

paramas.filterByCircularity = True
paramas.minCircularity = 0.1

detector = cv2.SimpleBlobDetector_create(paramas)

keypoints = detector.detect(image1)
blobdraw = cv2.drawKeypoints(image1,keypoints,image1,(255,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("sonic blob",blobdraw)

cv2.waitKey()
cv2.destroyAllWindows()


