import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2

#reading images

print("type a number between 1 to 5")
num_pair = input()
image_got_1 =  'foto'+num_pair+'A.jpg'
image_got_2 =  'foto'+num_pair+'B.jpg'

img1 = cv2.imread(image_got_1)
img2 = cv2.imread(image_got_2)

#1) Turning to black and white and saving it

grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

image_gray_a =  'gray'+num_pair+'A.jpg'
image_gray_b =  'gray'+num_pair+'B.jpg'
cv2.imwrite(image_gray_a, grayA)
cv2.imwrite(image_gray_b, grayB)

#2) SIFT

sift = cv2.xfeatures2d.SIFT_create()

imgAsift = np.copy(img1)
imgBsift = np.copy(img1)

kp_sift_A, des_sift_A = sift.detectAndCompute(grayA,None)
cv2.drawKeypoints(grayA, kp_sift_A, imgAsift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp_sift_B, des_sift_B = sift.detectAndCompute(grayB,None)
cv2.drawKeypoints(grayB, kp_sift_B, imgBsift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

image_sift_a =  './SIFT/sift'+num_pair+'A.jpg'
image_sift_b =  './SIFT/sift'+num_pair+'B.jpg'
cv2.imwrite(image_sift_a,imgAsift)
cv2.imwrite(image_sift_b,imgBsift)

#2) SURF

surf = cv2.xfeatures2d.SURF_create(400)

kp_surf_A, des_surf_a = surf.detectAndCompute(grayA,None)
imgASurf = cv2.drawKeypoints(grayA, kp_surf_A, None, (255,0,0),4)

kp_surf_B, des_surf_b = surf.detectAndCompute(grayB,None)
imgBSurf = cv2.drawKeypoints(grayB, kp_surf_B, None,(255,0,0),4)

image_surf_a =  './SURF/surf'+num_pair+'A.jpg'
image_surf_b =  './SURF/surf'+num_pair+'B.jpg'
cv2.imwrite(image_surf_a,imgASurf)
cv2.imwrite(image_surf_b,imgBSurf)

#2) BRIEF

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp_brief_a = star.detect(grayA,None)
kp_brief_a, des_brief_a = brief.compute(grayA, kp_brief_a)

kp_brief_b = star.detect(grayB,None)
kp_brief_b, des_brief_b = brief.compute(grayA, kp_brief_b)

#2) ORB

imgAOrb = np.copy(img1)
imgBOrb = np.copy(img1)
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

kp_orb_a = orb.detect(grayA,None)
kp_orb_a, des_orb_a = orb.compute(grayA, kp_orb_a)
cv2.drawKeypoints(grayA,kp_orb_a,imgAOrb,color=(0,255,0), flags=0)
image_orb_a =  './ORB/orb'+num_pair+'A.jpg'
cv2.imwrite(image_orb_a,imgAOrb)

kp_orb_b = orb.detect(grayB,None)
kp_orb_b, des_orb_b = orb.compute(grayB, kp_orb_b)
cv2.drawKeypoints(grayB,kp_orb_b,imgBOrb,color=(0,255,0), flags=0)
image_orb_b =  './ORB/orb'+num_pair+'B.jpg'
cv2.imwrite(image_orb_b,imgBOrb)

#3) Compute distance/similarities between each descriptor pair

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_sift = flann.knnMatch(des_sift_A, des_sift_B, k=2)
matches_surf = flann.knnMatch(des_surf_a, des_surf_b, k=2)

#4) Select best matching between each descriptor SIFT

good_points = []
ratio = 0.6
for m, n in matches_sift:
    if m.distance < ratio*n.distance:
        good_points.append(m)        
result_4 = cv2.drawMatches(grayA, kp_sift_A, grayB, kp_sift_B, good_points, None)
cv2.imwrite('./SIFT/imgAfter4_Sift.jpg',result_4)

#4) Select best matching between each descriptor SURF

good_points = []
ratio = 0.6
for m, n in matches_surf:
    if m.distance < ratio*n.distance:
        good_points.append(m)        
result_4 = cv2.drawMatches(grayA, kp_surf_A, grayB, kp_surf_B, good_points, None)
cv2.imwrite('./SURF/imgAfter4_Surf.jpg',result_4)

#5) 