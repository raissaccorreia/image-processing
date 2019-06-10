import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2

#reading images

img1 = cv2.imread('foto1A.jpg')
img2 = cv2.imread('foto1B.jpg')

#1) Turning to black and white

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#cv2.imwrite('./gray1A.jpg', gray1)
#cv2.imwrite('./gray1B.jpg', gray2)

#2) SIFT

sift = cv2.xfeatures2d.SIFT_create()


img1Asift = np.copy(img1)
img1Bsift = np.copy(img1)

kp_sift_A, des_sift_A = sift.detectAndCompute(gray1,None)
cv2.drawKeypoints(gray1,kp_sift_A,img1Asift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp_sift_B, des_sift_B = sift.detectAndCompute(gray2,None)
cv2.drawKeypoints(gray2,kp_sift_B,img1Bsift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('img1Asift.jpg',img1Asift)
cv2.imwrite('img1Bsift.jpg',img1Bsift)

#2) SURF
surf = cv2.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(gray1,None)
img1ASurf = cv2.drawKeypoints(gray1,kp,None,(255,0,0),4)
kp, des = surf.detectAndCompute(gray2,None)
img1BSurf = cv2.drawKeypoints(gray2,kp,None,(255,0,0),4)

cv2.imwrite('img1Asurf.jpg',img1ASurf)
cv2.imwrite('img1Bsurf.jpg',img1BSurf)

#2) BRIEF

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp = star.detect(img1,None)
kp, des = brief.compute(img1, kp)

kp = star.detect(img2,None)
kp, des = brief.compute(img2, kp)

#2) ORB

img1AOrb = np.copy(img1)
img1BOrb = np.copy(img1)
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

kp = orb.detect(img1,None)
kp, des = orb.compute(gray1, kp)
cv2.drawKeypoints(gray1,kp,img1AOrb,color=(0,255,0), flags=0)
cv2.imwrite('img1Aorb.jpg',img1AOrb)

kp = orb.detect(img2,None)
kp, des = orb.compute(gray2, kp)
cv2.drawKeypoints(gray2,kp,img1BOrb,color=(0,255,0), flags=0)
cv2.imwrite('img1Borb.jpg',img1BOrb)

#3) Compute distance/similarities between each descriptor pair

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_sift_A, des_sift_B, k=2)

#4) Select best matching between each descriptor

good_points = []
ratio = 0.6
for m, n in matches:
    if m.distance < ratio*n.distance:
        good_points.append(m)        
result_4 = cv2.drawMatches(gray1, kp_sift_A, gray2, kp_sift_B, good_points, None)
cv2.imwrite('imgAfter4.jpg',result_4)

#5) 