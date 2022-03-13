#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# 
# Aluno: Breno Moura de Abreu
# RA: 1561286
#===============================================================================

from re import L
import sys
from termios import B134
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 50

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    rows, cols, channels = img.shape

    # Not working right now :(
    #img_bin = np.where(img < threshold, 0, 1)

    # Percorre a imagem, alterando a intensidade pixel por pixel em cada canal
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                # Altera a intensidade do pixel de acordo com o threshold
                if img[row, col, channel] < threshold:
                    img[row, col, channel] = 0
                else:
                    img[row, col, channel] = 1

    return img

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo contémnte de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

    # Recebe as dimensões da matriz da imagem
    rows, cols, channels = img.shape

    # Cria uma nova matriz, recebendo -1 para pixels que fazem parte do background,
    # e 0 para pixels que fazem parte de algum blob
    matrix = np.where(img == 0, -1, 0)

    # Inicializa o rótulo de blobs
    label = 1

    # Cria a lista de componentes
    componentes = []

    # Percorre a imagem pixel por pixel
    for row in range(rows):
        for col in range(cols):
            # Se o valor do pixel na matriz é 0, singnifica que ele faz parte
            # de um blob mas ainda não foi visitado
            if matrix[row, col, 0] == 0:

                # Visita um blob utilizando a técnica flood fill recursiva
                # Será retornado o número de pixels total do blob
                # E as coordenadas das extremidades do retângulo que contém o blob
                n_pixels, T, L, B, R = flood_fill(label, matrix, row, col)

                # Se o número total de pixels, a largura ou altura do retângulo
                # é menor do que o definido, o blob não é considerado válido
                if (n_pixels >= n_pixels_min and 
                    B - T >= altura_min and 
                    R - L >= largura_min):

                    # Escreve as informações em um dicionário
                    info = {"label" : label,
                            "n_pixels" : n_pixels,
                            "T": T,
                            "L": L,
                            "B": B,
                            "R": R}

                    # Adiciona o blob à lista
                    componentes.append(info)

                    # Atualiza o rótulo
                    label = label + 1

    return componentes


def flood_fill(label, matrix, row, col):
    ''' Label: identificador do blob
        Matrix: a matriz que será percorrida para encontrar e retular os pixels
            que formam um blob
        Row: a coordenada Y do pixel
        Col: a coordenada X do pixel '''
        
    # Recebe o número de linhas e colunas da imagem
    rows, cols, _ = matrix.shape

    # Altera o label do pixel
    matrix[row, col, 0] = label

    # Variáveis que indicam as coordenadas do retângulo que contém o blob
    T1 = row
    L1 = col
    B1 = row
    R1 = col

    # Variável que recebe o número total de pixels do blob
    # Inicializada com 1 para representar o pixel que está sendo visitado
    n_pixels = 1

    # Visita os pixels adjascentes ao pixel atual em vizinhança-4
    # Verifica se estão dentro da imagem e se seu valor é 0, 
    # o que indica que um pixel faz parte de um blob mas não foi visitado
    # Os pixels são visitados recursivamente
    if row - 1 >= 0 and matrix[row - 1, col, 0] == 0:
        # Visita o pixel acima do pixel atual
        n_pixels_aux, T2, L2, B2, R2 = flood_fill(label, matrix, row - 1, col)
        # Soma o número de pixels atual com o valor recebido do pixel adjascente
        n_pixels = n_pixels + n_pixels_aux
        # Compara os valores dos conjuntos de coordenadas para encontrar
        # os valores das extremidades do blob
        T1, L1, B1, R1 = compare_pixels(T1, L1, B1, R1, T2, L2, B2, R2)

    if col - 1 >= 0 and matrix[row, col - 1, 0] == 0:
        # Visita o pixel a esquerda do pixel atual
        n_pixels_aux, T2, L2, B2, R2 = flood_fill(label, matrix, row, col - 1)
        n_pixels = n_pixels + n_pixels_aux
        T1, L1, B1, R1 = compare_pixels(T1, L1, B1, R1, T2, L2, B2, R2)

    if row + 1 < rows and matrix[row + 1, col, 0] == 0:
        # Visita o pixel abaixo do pixel atual
        n_pixels_aux, T2, L2, B2, R2 = flood_fill(label, matrix, row + 1, col)
        n_pixels = n_pixels + n_pixels_aux
        T1, L1, B1, R1 = compare_pixels(T1, L1, B1, R1, T2, L2, B2, R2)
    
    if col + 1 < cols and matrix[row, col + 1, 0] == 0:
        # Visita o pixel a direita do pixel atual
        n_pixels_aux, T2, L2, B2, R2 = flood_fill(label, matrix, row, col + 1)
        n_pixels = n_pixels + n_pixels_aux
        T1, L1, B1, R1 = compare_pixels(T1, L1, B1, R1, T2, L2, B2, R2)
        
    # Retorna o número de pixels do blob e as coordenadas que formam
    # o retângulo que contém o blob 
    return n_pixels, T1, L1, B1, R1


def compare_pixels(T1, L1, B1, R1, T2, L2, B2, R2):
    ''' Compara dois conjuntos de coordenadas e retorna os valores das 
        coordenadas que estão mais próximas das bordas da imagem
        Esta função é executada para encontrar as coordenadas mais extremas 
        do blob, permitindo que o retângulo que contém o blob seja desenhado '''

    T = min(T1, T2)
    L = min(L1, L2)
    B = max(B1, B2)
    R = max(R1, R2)

    return T, L, B, R

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
