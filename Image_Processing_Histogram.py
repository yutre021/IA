import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load a image
imagem = cv2.imread('userpath/',cv2.IMREAD_GRAYSCALE)

plt.imshow(imagem, cmap ="gray")
plt.axis('off')  # turn of ex..
plt.show()


#Calculate the histogram
histograma = cv2.calcHist([imagem], [0], None, [256], [0,256])

#Show the Histogram
plt.plot(histograma, color='gray')
plt.xlabel('Itensidade de Pixel')
plt.ylabel('Numero de Pixels')
plt.title('Histograma da Imagem')
plt.show()
