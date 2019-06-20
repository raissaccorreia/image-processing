import scipy
import numpy as np
import cv2

#Search to do it:
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
#https://docs.opencv.org/3.0-beta/modules/core/doc/clustering.html?highlight=kmeans

print("Choose an image: baboon, monalisa, peppers, watch")
source = input()
print("Choose a flag for centralization: PP or RANDOM")
flag_type = input()

img = cv2.imread(source+".png")
img_32 = np.float32(img)
k=16
attempts = 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


while (k <= 128):
    if(flag_type == "PP"):
        compactness,labels,centers = cv2.kmeans(img_32,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    if(flag_type == "RANDOM"):
        compactness,labels,centers = cv2.kmeans(img_32,k,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)    

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((img_32.shape))
    cv2.imwrite('out'+str(k)+'_levels'+flag_type+'.png', res2)
    k = k*2