import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import subprocess

#lendo imagem original, executando NOT e preparando copias
img = cv.imread('bitmap.pbm',0)
img = cv.bitwise_not(img)
imgAfter = np.copy(img)
imgAfter2 = np.copy(img)
imgAfter3 = np.copy(img)
imgAfter4 = np.copy(img)
imgAfter5 = np.copy(img)
imgAfter6 = np.copy(img)

#dilatacao do item 1
structuring = np.ones((1,100), np.uint8)
imgAfter = cv.dilate(img,structuring)
cv.imwrite('./imgAfter1.pbm', cv.bitwise_not(imgAfter))

#erosao do item 2
imgAfter2 = cv.erode(imgAfter,structuring)
cv.imwrite('./imgAfter2.pbm', cv.bitwise_not(imgAfter2))

#dilatacao do item 3
structuring_3 = np.ones((200,1), np.uint8)
imgAfter3 = cv.dilate(img,structuring_3)
cv.imwrite('./imgAfter3.pbm', cv.bitwise_not(imgAfter3))

#erosao do item 4
imgAfter4 = cv.erode(imgAfter3,structuring_3)
cv.imwrite('./imgAfter4.pbm', cv.bitwise_not(imgAfter4))

#AND do item 5
imgAfter5 = cv.bitwise_and(imgAfter2, imgAfter4)
cv.imwrite('./imgAfter5.pbm', cv.bitwise_not(imgAfter5))

#fechamento do item 6
structuring_6 = np.ones((1,30), np.uint8)
imgAfter6 = cv.morphologyEx(imgAfter5, cv.MORPH_CLOSE, structuring_6)
cv.imwrite('./imgAfter6.pbm', cv.bitwise_not(imgAfter6))

#chamar software de componentes conexos item 7
subprocess.call(['gcc', '-o', 'comp_conexos', 'comp_conexos.c', '-lm'])
subprocess.call(['./comp_conexos','./imgAfter6.pbm', './imgAfter7.pbm'])

#item 8

#saida para deixar claro que acabou
print("Done! Open the folder to check the results!")
