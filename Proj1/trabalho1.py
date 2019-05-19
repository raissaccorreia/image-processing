from scipy import ndimage, misc
import numpy.fft
import numpy
import matplotlib.pyplot as plt
import cv2

print("Escolha uma das seguintes imagens digitando as seguintes strings:")
print("city.png, baboon.png, butterfly.png,house.png,seagull.png")
entrada = input()
print("Escolha o sigma")
fator_sigma = float(input())

fig, (ax1, ax2) = plt.subplots(1, 2) #criando o plot com antes e depois
plt.gray() #em escala cinza
img = cv2.imread(entrada,0) #adicionando a imagem

input_ = numpy.fft.fft2(img) #aplicando fourier na imagem
result = ndimage.fourier_gaussian(input_, sigma=fator_sigma) #filtro gaussiano com axis=-1
result = numpy.fft.ifft2(result) #revertendo fourier
ax1.imshow(img) #plotando
ax2.imshow(result.real) #apenas parte real
plt.show() #exibicao

#kernel 1
h1 = numpy.array([[0,0,-1,0,0], [0,-1,-2,-1,0], [-1,-2,16,-2,-1], [0,0,-1,0,0],[0,-1,-2,-1,0]])
#kernel 2 e a multiplicacao pelo escalar
h2 = numpy.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
h2 = 0.00390625*h2
#kernel 3
h3 = numpy.array([[-1,0,1], [-2,0,2], [-1,0,1]])
#kernel 4
h4 = numpy.array([[-1,-2,-1],[0,0,0],[1,2,1]])
#calculo e arredondamento do kernel 5
h5 = numpy.sqrt(numpy.square(h3) + numpy.square(h4))
h5 = numpy.around(h5,decimals=0)

#img para numpy array
img = cv2.imread(entrada)
#aplicando convolucao do cv2
dst1 = cv2.filter2D(img, -1, h1)
#escrevendo como arquivo png
cv2.imwrite("./filtered/filtered1.png", dst1)

dst2 = cv2.filter2D(img, -1, h2)
cv2.imwrite("./filtered/filtered2.png", dst2)

dst3 = cv2.filter2D(img, -1, h3)
cv2.imwrite("./filtered/filtered3.png", dst3)

dst4 = cv2.filter2D(img, -1, h4)
cv2.imwrite("./filtered/filtered4.png", dst4)

dst5 = cv2.filter2D(img, -1, h5)
#cv2.normalize(dst5,dst5,0,1,2,-1) tentativa de normalizacao de dst5 sem sucesso
cv2.imwrite("./filtered/filtered5.png", dst5)