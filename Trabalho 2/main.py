#===============================================================================
# Processamento Digital de Imagens
# Trabalho 2 - Blur
# Autor: Breno Moura de Abreu
# RA: 1561286
# Data: 2022-03-27
#===============================================================================

#----------------------------------LEIA-ME--------------------------------------
#
# 
#
#-------------------------------------------------------------------------------

#===============================================================================

import sys
import cv2
import numpy as np

#===============================================================================

INPUT_IMAGE = "b01 - Original.bmp"
BLUR_TYPE = "ing"
WINDOW_HEIGHT = 3
WINDOW_WIDTH = 3

#===============================================================================

def run_naive_method(img, wnd_width, wnd_height):
    n_pixels = wnd_height * wnd_width
    rows, cols, channels = img.shape

    half_height = int(wnd_height / 2)
    half_width = int(wnd_width / 2)

    blured_img = np.zeros((rows, cols, channels), np.float32)

    for y in range(rows):
        for x in range(cols):
            for c in range(channels):
                if (y - half_height >= 0    and 
                    y + half_height < rows  and
                    x - half_width >= 0     and
                    x + half_width < cols):
                    total = 0
                    for h in range(y - half_height, y + half_height + 1):
                        for w in range(x - half_width, x + half_width + 1):
                            total += img[h, w, c]
                    
                    blured_img[y, x, c] = total / n_pixels
            
                else:
                    blured_img[y, x, c] = img[y, x, c]
    
    return blured_img

#===============================================================================

def run_separable_filter_method(img, wnd_width, wnd_height):

    rows, cols, channels = img.shape

    half_height = int(wnd_height / 2)
    half_width = int(wnd_width / 2)

    buffer = np.zeros((rows, cols, channels), np.float32)
    blured_img = np.zeros((rows, cols, channels), np.float32)

    for y in range(rows):
        for x in range(cols):
            for c in range(channels):
                if (x - half_width >= 0     and
                    x + half_width < cols):
                    total = 0
                    for w in range(x - half_width, x + half_width + 1):
                        total += img[y, w, c]
                    
                    buffer[y, x, c] = total / wnd_width
            
                else:
                    buffer[y, x, c] = img[y, x, c]

    for y in range(rows):
        for x in range(cols):
            for c in range(channels):
                if (y - half_height >= 0    and 
                    y + half_height < rows  and
                    x - half_width >= 0     and
                    x + half_width < cols):
                    total = 0
                    for h in range(y - half_height, y + half_height + 1):
                        total += buffer[h, x, c]
                    
                    blured_img[y, x, c] = total / wnd_height
            
                else:
                    blured_img[y, x, c] = img[y, x, c]
    
    return blured_img

#===============================================================================

def run_integral_method(img, wnd_width, wnd_height):

    n_pixels = wnd_height * wnd_width
    rows, cols, channels = img.shape

    half_height = int(wnd_height / 2)
    half_width = int(wnd_width / 2)

    blured_img = np.zeros((rows, cols, channels), np.float32)
    integral_img = get_integral_img(img)

    for y in range(rows):
        for x in range(cols):
            for c in range(channels):
                B = y + half_height
                R = x + half_width
                T = y - half_height - 1
                L = x - half_width - 1

                if B > rows - 1: B = rows - 1
                if R > cols - 1: R = cols - 1
                if T < 0: T = 0
                if L < 0: L = 0

                blured_img[y, x, c] = ((integral_img[B, R, c] - 
                                        integral_img[B, L, c] -
                                        integral_img[T, R, c] +
                                        integral_img[T, L, c]
                                        ) / n_pixels)
    
    return blured_img

#-------------------------------------------------------------------------------

def get_integral_img(img):
    rows, cols, channels = img.shape

    integral_img = np.zeros((rows, cols, channels), np.float32)

    for y in range(rows):
        for x in range(cols):
            for c in range(channels):
                if x != 0:
                    integral_img[y, x, c] = img[y, x, c] + integral_img[y, x - 1, c]
                else:
                    integral_img[y, x, c] = img[y, x, c]

    for y in range(1, rows):
        for x in range(cols):
            for c in range(channels):
                integral_img[y, x, c] = integral_img[y, x, c] + integral_img[y - 1, x, c]
    
    return integral_img

#===============================================================================

def main():

    if len(sys.argv) != 5:
        print("Por favor, informe o nome do arquivo de imagem e o tipo de processamento.\n"
              "Veja como informar os argumentos na seção LEIA-ME no início do arquivo contendo o código.")
    else:
        INPUT_IMAGE = sys.argv[1]
        WINDOW_WIDTH = int(sys.argv[2])
        WINDOW_HEIGHT = int(sys.argv[3])
        BLUR_TYPE = sys.argv[4]
        
        img = cv2.imread(INPUT_IMAGE)

        if img is None:
            print('Erro ao abrir a imagem.\n')
            sys.exit ()

        img = img.reshape((img.shape [0], img.shape [1], 3))    
        img = img.astype(np.float32) / 255

        if BLUR_TYPE == '-ing':
            blured_img = run_naive_method(img, WINDOW_WIDTH, WINDOW_HEIGHT)
        elif BLUR_TYPE == '-sep':
            blured_img = run_separable_filter_method(img, WINDOW_WIDTH, WINDOW_HEIGHT)
        elif BLUR_TYPE == '-int':
            blured_img = run_integral_method(img, WINDOW_WIDTH, WINDOW_HEIGHT)
        else:
            print('O tipo de processamento não foi informado corretamente!')
        
        cv2.imshow ('Blured Image', blured_img)
        cv2.imwrite ('Blured Image.png', blured_img*255)
        cv2.waitKey ()
        cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()