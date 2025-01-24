import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carregar a imagem
imagem = cv2.imread('C:/Users/ycar2/OneDrive/Área de Trabalho/2024/1_Semestre/Processamento/pexels_03.webp',cv2.IMREAD_GRAYSCALE)

plt.imshow(imagem, cmap ="gray")
plt.axis('off')  # Desligar os eixos
plt.show()


#calcula o histograma
histograma = cv2.calcHist([imagem], [0], None, [256], [0,256])

#Mostrar o Histograma
plt.plot(histograma, color='gray')
plt.xlabel('Itensidade de Pixel')
plt.ylabel('Numero de Pixels')
plt.title('Histograma da Imagem')
plt.show()