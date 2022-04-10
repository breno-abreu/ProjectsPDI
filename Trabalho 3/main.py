#===============================================================================
# Processamento Digital de Imagens
# Trabalho 3 - Bloom
# Autor: Breno Moura de Abreu
# RA: 1561286
# Data: 2022-04-09
#===============================================================================

import cv2
import numpy as np

#===============================================================================

# Constantes usadas nos algoritmos

# Caminho da imagem que será carregada
IMG_NAME = "img7.jpg"

# Threshold para o bright-pass
THRESHOLD = 0.7

# Quantidade de imagens borradas que serão somadas no box blur
BOX_BLUR_N_SUMS = 4

# Dimensões do kernel do box blur
BOX_BLUR_KERNEL = (15, 15)

# Quantidade de vezes que a mesma imagem será borrada no box blur
# Usado para achar a aproximação do gaussian blur com o box blur
BOX_BLUR_N_BLURS = 5

# Quantidade de imagens borradas que serão somadas no gaussian blur
GAUSS_BLUR_N_SUMS = 4

# Valor do alfa do gaussian blur
GAUSS_BLUR_ALPHA = 10

# Coeficiente que será multiplicado pela imagem original ao mesclá-lo com a máscara
ORIGINAL_IMG_COEF = 0.8

# Coeficiente que será multiplicado pela máscara ao mesclá-lo com a iamgem original
MASK_COEF = 0.2

#===============================================================================

def get_mask(img, threshold):
    ''' Encontra os pixels mais luminosos da imagem
        Retorna uma imagem contendo apenas esses pixels '''

    # Encontra as dimensões da imagem original
    rows, cols, channels = img.shape
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Transforma a imagem de RGB para HSL
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Inicializa a matriz da máscara
    mask = np.zeros((rows, cols, channels), np.float32)

    # Percorre cada pixel da imagem original
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):

                # Se o valor de luminância for maior que o threshold,
                # Guarda o pixel na máscara
                if hls[row, col, 1] > threshold:
                    mask[row, col, channel] = img[row, col, channel]
                
                # Caso contrário o pixel fica preto
                else:
                    mask[row, col, channel] = 0
    
    cv2.imwrite('Mask-NoBlur.png', mask * 255)

    return mask

def box_blur(mask, n_sums, kernel, n_blurs):
    ''' Aplica o box blur na máscara '''

    # Encontra as dimensões da máscara
    rows, cols, channels = mask.shape

    # Inicializa a matriz da máscara borrada
    blurred_mask = np.zeros((rows, cols, channels), np.float32)

    buffer = mask

    # Cria n_sums imagens borradas e soma seus valores
    for i in range(n_sums):
        for j in range(n_blurs):

            # Para tentar criar uma aproximação com o gaussian blur,
            # Borra a mesma imagem n_blurs vezes
            buffer = cv2.blur(buffer, kernel)

        # Soma todas as imagens borradas criadas
        # Cada imagem nova terá uma janela maior que a janela utilizada
        # na imagem anterior
        blurred_mask += buffer
    
    cv2.imwrite('Mask-BoxBlur.png', blurred_mask * 255)
    
    return blurred_mask

def gauss_blur(mask, n_sums, alpha):
    ''' Aplica o gaussian blur na máscara '''
    
    # Encontra as dimensões da máscara
    rows, cols, channels = mask.shape

    # Inicializa a matriz da máscara borrada
    blurred_mask = np.zeros((rows, cols, channels), np.float32)
    
    # Cria n_sums imagens borradas e soma seus valores
    for i in range(n_sums):
        blurred_mask += cv2.GaussianBlur(mask, (-1, -1), alpha)

        # Atualiza o valor de alpha para aumentar o tamanho da janela
        alpha *= 2
    
    cv2.imwrite('Mask-GaussBlur.png', blurred_mask * 255)

    return blurred_mask

def merge(img, img_coef, mask, mask_coef, img_name):
    ''' Mescla a imagem original e a máscara
        utilizando coeficientes diferentes para cada imagem '''

    merged_img = img * img_coef + mask * mask_coef
    cv2.imwrite(img_name, merged_img * 255)

def main():

    # Carrega a imagem
    img = cv2.imread(IMG_NAME)

    # Normaliza os valores da imagem para valores entre 0 e 1
    img = img.reshape((img.shape [0], img.shape [1], 3))    
    img = img.astype(np.float32) / 255

    # Cria a máscara que contém os pixels com maior luminância
    mask = get_mask(img, THRESHOLD)

    # Aplica o box blur à máscara e o mescla com a imagem original
    blurred_mask = box_blur(mask, BOX_BLUR_N_SUMS, BOX_BLUR_KERNEL, BOX_BLUR_N_BLURS)
    merge(img, ORIGINAL_IMG_COEF, blurred_mask, MASK_COEF, "Merged-BoxBlur.png")

    # Aplica o gaussian blur à máscara e o mescla com a imagem original
    blurred_mask = gauss_blur(mask, GAUSS_BLUR_N_SUMS, GAUSS_BLUR_ALPHA)
    merge(img, ORIGINAL_IMG_COEF, blurred_mask, MASK_COEF, "Merged-GaussBlur.png")

if __name__ == "__main__":
    main()