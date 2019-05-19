import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2

#definindo matriz bayer
bayer = np.array([[0,12,3,15], [8,4,11,7], [2,14,1,13], [10,6,9,5]])

print("this might take up to 40 seconds!")

inputs = ["baboon.pgm","fiducial.pgm","monarch.pgm","peppers.pgm","retina.pgm","sonnet.pgm","wedge.pgm"]

for a in range(0,len(inputs)): #to all images available
    entrada = inputs[a]
    img = cv2.imread(entrada,0) #input
    height, width = img.shape #getting height and width
    imgBayer = np.copy(img) #copying to make bayer
    imgFloyd = np.copy(img) #copying to make floyd-steinberg

#doind bayer
    for i in range(height):
        for j in range(width):
            gray_level = imgBayer[i][j]
            normalized_gray = (gray_level/255)*9
            if normalized_gray < bayer[i%4][j%4]:
                imgBayer[i][j] = 0
            else:
                imgBayer[i][j] = 255

    path = "./filtered2/" + entrada[:-4] + "_after_bayer.pbm" #saving to this path
    cv2.imwrite(path, imgBayer)

#doind floyd-steinberg

    for i in range(height):
        if i % 2 == 0: #even lines from left to right
            for j in range(0, width):
                gray_level = imgFloyd[i][j]
                if gray_level < 127:
                    new_level = 0
                else:
                    new_level = 255
                imgFloyd[i][j] = new_level
                amount_error = (gray_level - new_level)
                if i == height-1 or j == width-1:
                    continue
                imgFloyd[i+1][j]   = imgFloyd[i+1][j]   + amount_error*(7/16)
                imgFloyd[i-1][j+1] = imgFloyd[i-1][j+1] + amount_error*(3/16)
                imgFloyd[i][j+1]   = imgFloyd[i][j+1]   + amount_error*(5/16)
                imgFloyd[i+1][j+1] = imgFloyd[i+1][j+1] + amount_error*(1/16)
        else:
            for j in range(width, 0): #odd lines from right to left
                gray_level = imgFloyd[i][j]
                if gray_level < 127:
                    new_level = 0
                else:
                    new_level = 255
                imgFloyd[i][j] = new_level
                amount_error = (gray_level - new_level)
                if i == height-1 or j == width-1:
                    continue
                imgFloyd[i+1][j]   = imgFloyd[i+1][j]   + amount_error*(7/16)
                imgFloyd[i-1][j+1] = imgFloyd[i-1][j+1] + amount_error*(3/16)
                imgFloyd[i][j+1]   = imgFloyd[i][j+1]   + amount_error*(5/16)
                imgFloyd[i+1][j+1] = imgFloyd[i+1][j+1] + amount_error*(1/16)

    path2 = "./filtered2/" + entrada[:-4] + "_after_floyd.pbm" #saving to this path
    cv2.imwrite(path2, imgFloyd)

print("Done! Open filtered2 folder to check the results!")