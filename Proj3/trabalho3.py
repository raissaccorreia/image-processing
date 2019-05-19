import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import subprocess

#print("this might take up to 40 seconds!")

#lendo imagem original
img = cv.imread('bitmap.pbm',0)
img = cv.bitwise_not(img)
height, width = img.shape
imgAfter = np.copy(img)
imgAfter2 = np.copy(img)

#criando elemento estruturante
structuring = np.ones((1,100), np.uint8)

#dilatacao do item 1
cv.dilate(img,structuring,imgAfter)
cv.imwrite('./imgAfter1.pbm', imgAfter)

#erosao do item 2
cv.erode(imgAfter,structuring,imgAfter)
cv.imwrite('./imgAfter2.pbm', imgAfter)

#dilatacao do item 3
structuring_3 = np.ones((200,1), np.uint8)
cv.dilate(img,structuring_3,imgAfter2)
cv.imwrite('./imgAfter3.pbm', imgAfter2)

#erosao do item 4
cv.erode(imgAfter2,structuring_3,imgAfter2)
cv.imwrite('./imgAfter4.pbm', imgAfter2)

#AND do item 5
imgAfter2 = cv.bitwise_and(imgAfter, imgAfter2)
cv.imwrite('./imgAfter5.pbm', imgAfter2)

#fechamento do item 6
structuring_6 = np.ones((1,30), np.uint8)

imgAfter2 = cv.morphologyEx(imgAfter2, cv.MORPH_CLOSE, structuring_6)
cv.imwrite('./imgAfter6.pbm', imgAfter2)

#chamar software de componentes conexos item 7
subprocess.call(['gcc', '-o', 'comp_conexos', 'comp_conexos.c', '-lm'])
subprocess.call(['./comp_conexos','./imgAfter6.pbm', './imgAfter7.pbm'])

#item 8
output = cv.ConnectedComponentsTypes(imgAfter2)

#saida
print("Done! Open filtered2 folder to check the results!")

#problemas em usar o resultado de uma para a outra. Discussao essencial la pras 10h do dia 19