#===============================================================================
# Processamento Digital de Imagens
# Trabalho 2 - Blur
# Autor: Breno Moura de Abreu
# RA: 1561286
# Data: 2022-03-27
#===============================================================================

#----------------------------------LEIA-ME--------------------------------------
#
# Para executar o programa é necessário informar os seguintes argumentos no CLI
# python main.py [nome do arquivo de imagem] [comprimento da janela] [altura da janela] [tipo de processamento]
# 
# Os tipos de processamento são:
#   Algoritmo ingênuo: -ing
#   Algoritmo de filtro separável: -sep
#   Algoritmo com imagens integrais: -int
#
# Exemplo de comando no CLI:
# python main.py "a01 - Original.bmp" 3 13 -int
#
#-------------------------------------------------------------------------------

#===============================================================================

import sys
import cv2
import numpy as np
import time

#===============================================================================

# Nome da imagem de entrada
INPUT_IMAGE = "b01 - Original.bmp"

# Tipo de processamento da imagem
BLUR_TYPE = "-ing"

# Comprimento da janela em pixels
WINDOW_WIDTH = 3

# Altura da janela em pixels
WINDOW_HEIGHT = 3

#===============================================================================

def run_naive_method(img, wnd_width, wnd_height):
    ''' Executa o algoritmo ingênuo '''

    # Armazena o total de pixels dentro da janela
    n_pixels = wnd_height * wnd_width

    # Armazena a quantidade de linhas, colunas e canais da imagem
    rows, cols, channels = img.shape

    # Armazena a metade do valor da altura e do comprimento da janela
    half_height = int(wnd_height / 2)
    half_width = int(wnd_width / 2)

    # Inicializa a matriz da imagem borrada
    blured_img = np.zeros((rows, cols, channels), np.float32)

    # Percorre pixel a pixel
    for y in range(rows):
        for x in range(cols):
            for c in range(channels):

                # Se a janela de um pixel está dentro da imagem,
                # calcula a média dos pixels da janela
                if (y - half_height >= 0    and 
                    y + half_height < rows  and
                    x - half_width >= 0     and
                    x + half_width < cols):
                    total = 0

                    # Percorre todos os pixels dentro de uma janela
                    # e soma seus valores
                    for h in range(y - half_height, y + half_height + 1):
                        for w in range(x - half_width, x + half_width + 1):
                            total += img[h, w, c]
                    
                    # Calcula a média dos valores dentro da janela
                    # e armazena o novo valor na matriz da imagem borrada
                    blured_img[y, x, c] = total / n_pixels
            
                # Se a janela de um pixel está fora da imagem, 
                # apenas armazena o valor original
                else:
                    blured_img[y, x, c] = img[y, x, c]
    
    return blured_img

#===============================================================================

def run_separable_filter_method(img, wnd_width, wnd_height):
    ''' Executa o algoritmo de filtro separável '''

    # Armazena a quantidade de linhas, colunas e canais da imagem
    rows, cols, channels = img.shape

    # Armazena a metade do valor da altura e do comprimento da janela
    half_height = int(wnd_height / 2)
    half_width = int(wnd_width / 2)

    # Inicializa a matriz de buffer e a matriz da imagem borrada
    buffer = np.zeros((rows, cols, channels), np.float32)
    blured_img = np.zeros((rows, cols, channels), np.float32)

    # Percorre pixel a pixel
    for y in range(rows):
        for x in range(cols):
            for c in range(channels):

                # Se a janela está dentro da imagem,
                # soma os valores na horizontal
                if (x - half_width >= 0     and
                    x + half_width < cols):
                    total = 0

                    # Soma os valores da parte horizontal da janela
                    for w in range(x - half_width, x + half_width + 1):
                        total += img[y, w, c]
                    
                    # Calcula a média dos valores
                    buffer[y, x, c] = total / wnd_width
            
                # Se a janela está fora da imagem, apenas copia o valor original
                else:
                    buffer[y, x, c] = img[y, x, c]

    # Percorre pixel a pixel
    for y in range(rows):
        for x in range(cols):
            for c in range(channels):

                # Se a janela está dentro da imagem, soma os valores na vertical
                if (y - half_height >= 0    and 
                    y + half_height < rows  and
                    x - half_width >= 0     and
                    x + half_width < cols):
                    total = 0

                    # Faz a soma dos valores na vertical
                    for h in range(y - half_height, y + half_height + 1):
                        total += buffer[h, x, c]
                    
                    # Calcula a média dos valores que estão dentro da janela
                    blured_img[y, x, c] = total / wnd_height
            
                # Se a janela está fora da imagem, apenas copia o valor original
                else:
                    blured_img[y, x, c] = img[y, x, c]
    
    return blured_img

#===============================================================================

def run_integral_method(img, wnd_width, wnd_height):
    ''' Executa o algoritmo utilizando imagens integrais '''

    # Armazena o total de pixels dentro da janela
    n_pixels = wnd_height * wnd_width

    # Armazena a quantidade de linhas, colunas e canais da imagem
    rows, cols, channels = img.shape

    # Armazena a metade do valor da altura e do comprimento da janela
    half_height = int(wnd_height / 2)
    half_width = int(wnd_width / 2)

    # Inicializa a matriz da imagem borrada
    blured_img = np.zeros((rows, cols, channels), np.float32)

    # Cria a imagem integral
    integral_img = get_integral_img(img)

    # Percorre pixel a pixel
    for y in range(rows):
        for x in range(cols):
            for c in range(channels):

                # Armazena a posição dos pixel bottom, right, top e left da janela
                B = y + half_height
                R = x + half_width
                T = y - half_height - 1
                L = x - half_width - 1

                # Se um dos pixels da janela estiver fora da imagem, 
                # atualiza seu valor para que esteja dentro da imagem
                if B > rows - 1: B = rows - 1
                if R > cols - 1: R = cols - 1
                if T < 0: T = 0
                if L < 0: L = 0

                # Faz a soma dos valores dos quatro pixels da extremidade da janela
                # e encontra a média do pixel central
                blured_img[y, x, c] = ((integral_img[B, R, c] - 
                                        integral_img[B, L, c] -
                                        integral_img[T, R, c] +
                                        integral_img[T, L, c]
                                        ) / n_pixels)
    
    return blured_img

#-------------------------------------------------------------------------------

def get_integral_img(img):
    ''' Cria a imagem integral '''

    # Armazena a quantidade de linhas, colunas e canais da imagem
    rows, cols, channels = img.shape

    # Inicializa a imagem integral
    integral_img = np.zeros((rows, cols, channels), np.float32)

    # Percore pixel a pixel
    for y in range(rows):
        for x in range(cols):
            for c in range(channels):

                # Soma o valor de um pixel com o valor do pixel à esquerda
                if x != 0:
                    integral_img[y, x, c] = img[y, x, c] + integral_img[y, x - 1, c]
                else:
                    integral_img[y, x, c] = img[y, x, c]

    # ercorre pixel a pixel
    for y in range(1, rows):
        for x in range(cols):
            for c in range(channels):

                # Soma o valor de um pixel com o valor do pixel acima
                integral_img[y, x, c] = integral_img[y, x, c] + integral_img[y - 1, x, c]
    
    return integral_img

#===============================================================================

def main():

    if len(sys.argv) != 5:
        print("Por favor, informe o nome do arquivo de imagem e o tipo de processamento.\n"
              "Veja como informar os argumentos na seção LEIA-ME no início do arquivo contendo o código.")
    else:
        
        # Armazena os valores recebidos como argumento pelo CLI
        INPUT_IMAGE = sys.argv[1]
        WINDOW_WIDTH = int(sys.argv[2])
        WINDOW_HEIGHT = int(sys.argv[3])
        BLUR_TYPE = sys.argv[4]

        # Armazena o timestamp de início
        start = time.time()
        
        # Carrega a imagem
        img = cv2.imread(INPUT_IMAGE)

        # Verifica se a imagem pôde ser carregada
        if img is None:
            print('Erro ao abrir a imagem.\n')
            sys.exit ()

        # Transforma os valroes da imagem para valores entre 0 e 1
        img = img.reshape((img.shape [0], img.shape [1], 3))    
        img = img.astype(np.float32) / 255

        # Executa o algoritmo ingênuo
        if BLUR_TYPE == '-ing':
            blured_img = run_naive_method(img, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Executa o algoritmo de filtro separável
        elif BLUR_TYPE == '-sep':
            blured_img = run_separable_filter_method(img, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Executa o algoritmo com iamgens integrais
        elif BLUR_TYPE == '-int':
            blured_img = run_integral_method(img, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        else:
            print('O tipo de processamento não foi informado corretamente!')
        
        # Calcula o tempo de execução total
        finish = time.time() - start
        finish = round(finish, 2)
        print("Tempo de execução: " + str(finish) + " segundos")

        # Mostra a imagem borrada
        cv2.imshow ('Blured Image', blured_img)

        # Salva a imagem borrada
        cv2.imwrite ('Blured Image.png', blured_img*255)

        # Fecha a imagem que foi aberta
        cv2.waitKey ()
        cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()