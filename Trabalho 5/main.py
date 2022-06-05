#===============================================================================
# Processamento Digital de Imagens
# Trabalho 5 - Chroma Key
# Autor: Breno Moura de Abreu
# RA: 1561286
# Data: 2022-05-23
#===============================================================================

#=================================LEIA ME=======================================
'''
Olá, professor. Infelizmente, novamente, o programa não funciona corretamente,
apesar de funcionar em certas situações. Não funciona para todas as imagens
e apresenta defeitos nas imagens que funciona.

Porém, não apenas me falta tempo que preciso investir em outros trabalhos
mas também esgotei as ideias do que poderia funcionar. Tentei algumas técnicas
diferentes como encontrar as bordas da imagem que dividem o foreground do
background mas minhas tentativas de tentar tratá-la não funcionaram muito bem.

Com certeza ter me consultado com o senhor teria ajudado bastante, mas como
falei não tive tanto tempo para conseguir fazer algo melhor.

O programa funciona da seguinte forma:
1. Encontra o background ao medir a distância entre a cor verde e o pixel
que está sendo analisado. Para tal é calculada a distância euclidiana entre
os três canais de um pixel. Se a distância for menor que uma constante, 
ele considera como sendo parte do background.

2. Realizar uma erosão após fazer a binarização do background e do foreground.
Erode os objetos do foreground para melhorar a qualidade da fronteira.
Levando em conta a etapa 3 ele acaba mostrando bons resultados para certas áreas
mas resultados ruins para outras.

3. Diminui o contraste de um pixel do background caso haja alguma sombra na
imagem. Dessa forma as sombras podem ser mostradas na imagem final.
Áreas mais escuras do background também ficam mais escuras na imagem final.
'''
#===============================================================================

import cv2 as cv
import numpy as np
import math

#===============================================================================

# Constante para definir a distância máxima entre cores de pixel que
# pertence ao background
BIN_THRESH = 150

# Kernel usado na erosão
EROSION_KERNEL = (2, 5)

# Valor da cor verde
GREEN = (0, 255, 0)

def main():

    # Carrega a imagem do foreground
    img = cv.imread("2.bmp")

    # Carrega a imagem do background
    img_bg = cv.imread("back.bmp")
    
    # Encontra as dimensões da imagem de foreground
    rows, cols, chs = img.shape

    # Redimensiona a imagem do background de acordo com as dimensões
    # da imagem do foreground
    img_bg = cv.resize(img_bg, (cols, rows), interpolation=cv.INTER_AREA)

    # Cria uma imagem em grayscale da iamgem do foreground
    img_gs = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gs = img_gs.astype(np.float32) / 255  
    img_gs = img_gs.reshape((img_gs.shape[0], img.shape[1], 1)) 

    # Cria a iamgem que será binarizada
    img_bin = img_gs.copy()

    # Valor que armazena o valor do canal verde com maior valor do background
    green_max = 0

    # Binariza a imagem e encontra o valor do canal verde com maior valor
    for y in range(rows):
        for x in range(cols):
            # encontra a distância euclidiana entre as cores de dois pontos
            dist = get_distance(GREEN, (img[y, x, 0], img[y, x, 1], img[y, x, 2]))

            # Se a distância for menor que a constante, faz parte do background
            if dist < BIN_THRESH:
                img_bin[y, x] = 0

                # Encontra o valor do canal verde com maior valor
                if img[y, x, 1] > green_max:
                    green_max = int(img[y, x, 1])
            
            # Caso contrário faz parte do foreground
            else:
                img_bin[y, x] = 1
            
    
    cv.imshow('Mask-NoBlur.png', img_bin)
    cv.waitKey()

    #dx = cv.Sobel(img_bin, cv.CV_32F, 1, 0, ksize=3)
    #dy = cv.Sobel(img_bin, cv.CV_32F, 0, 1, ksize=3)
    #img_mag = cv.magnitude(dx, dy)

    # Faz a erosão da imagem binarizada
    ero_kernel = np.ones(EROSION_KERNEL)
    img_ero = cv.erode(img_bin, ero_kernel, iterations=1)

    cv.imshow('Mask-NoBlur.png', img_ero)
    cv.waitKey()

    # Imagem final
    img_merged = img.copy()

    # Faz a fusão das imagens do foreground e do background
    for y in range(rows):
        for x in range(cols):
            for c in range(chs):

                # Se faz parte do background
                if img_ero[y, x] == 0:

                    # Calcula um peso que será aplicado para áreas mais escuras
                    # do background. Altera o contraste
                    coef = int(img[y, x, 1]) / green_max
                    img_merged[y, x, c] = img_bg[y, x, c] * coef
                
                # Se faz parte do foreground
                else:
                    img_merged[y, x, c] = img[y, x, c]
    
    cv.imshow('Mask-NoBlur.png', img_merged)
    cv.waitKey()

# Encontra a distância euclidiana entre as cores de dois pixels
def get_distance(p0, p1):
    return math.sqrt(pow(p1[0] - p0[0], 2) + pow(p1[1] - p0[1], 2) + pow(p1[2] - p0[2], 2))

if __name__ == "__main__":
    main()