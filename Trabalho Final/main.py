import cv2 as cv
import numpy as np

def main():
    img = cv.imread("E.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows, cols = img.shape

    for y in range(rows):
        for x in range(cols):
            if img[y, x] > 100:
                img[y, x] = 255
            else:
                img[y, x] = 0
    
    totals = []

    for y in range(rows):
        total = 0
        for x in range(cols):
            if img[y, x] == 0:
                total = total + 1
        totals.append(total)
    
    horizontalBars = {}
    count = 0
    isBar = False
    for i in range(len(totals)):
        if not isBar and totals[i] / cols >= 0.5:
            isBar = True
            count = count + 1
            if totals[i] / cols >= 0.9:
                if i < cols * 0.3:
                    horizontalBars[str(count)] = "Top"
                elif i >= cols * 0.3 and i < cols * 0.7:
                    horizontalBars[str(count)] = "Middle"
                else:
                    horizontalBars[str(count)] = "Bottom"
            else:
                print(cols * 0.6)
                if i < cols * 0.3:
                    horizontalBars[str(count)] = "TopS"
                elif i >= cols * 0.3 and i < cols * 0.7:
                    horizontalBars[str(count)] = "MiddleS"
                else:
                    horizontalBars[str(count)] = "BottomS"

        elif isBar and totals[i] / cols < 0.5:
            isBar = False

    totals.clear()

    for x in range(cols):
        total = 0
        for y in range(rows):
            if img[y, x] == 0:
                total = total + 1
        totals.append(total)

    verticalBars = {}
    count = 0
    isBar = False
    for i in range(len(totals)):
        if not isBar and totals[i] / rows >= 0.9:
            isBar = True
            count = count + 1
            if i < rows * 0.3:
                verticalBars[str(count)] = "Left"
            elif i >= rows * 0.3 and i < rows * 0.6:
                verticalBars[str(count)] = "Middle"
            else:
                verticalBars[str(count)] = "Right"
        elif isBar and totals[i] / rows < 0.9:
            isBar = False
    
    print(verticalBars)
    print(horizontalBars)

def label_blobs (img, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
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
                if n_pixels >= n_pixels_min:

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
        Matrix: a matriz que será percorrida para encontrar e rotular os pixels
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




if __name__ == "__main__":
    main()