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

Par??metros: img: imagem de entrada E sa??da.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item ?? um vetor associativo (dictionary)
com os seguintes campos:

'label': r??tulo do componente.
'n_pixels': n??mero de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do ret??ngulo cont??mnte de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta fun????o.
    # Use a abordagem com flood fill recursivo.

    # Recebe as dimens??es da matriz da imagem
    rows, cols, channels = img.shape

    # Cria uma nova matriz, recebendo -1 para pixels que fazem parte do background,
    # e 0 para pixels que fazem parte de algum blob
    matrix = np.where(img == 0, -1, 0)

    # Inicializa o r??tulo de blobs
    label = 1

    # Cria a lista de componentes
    componentes = []

    # Percorre a imagem pixel por pixel
    for row in range(rows):
        for col in range(cols):
            # Se o valor do pixel na matriz ?? 0, singnifica que ele faz parte
            # de um blob mas ainda n??o foi visitado
            if matrix[row, col, 0] == 0:

                # Visita um blob utilizando a t??cnica flood fill recursiva
                # Ser?? retornado o n??mero de pixels total do blob
                # E as coordenadas das extremidades do ret??ngulo que cont??m o blob
                n_pixels, T, L, B, R = flood_fill(label, matrix, row, col)

                # Se o n??mero total de pixels, a largura ou altura do ret??ngulo
                # ?? menor do que o definido, o blob n??o ?? considerado v??lido
                if n_pixels >= n_pixels_min:

                    # Escreve as informa????es em um dicion??rio
                    info = {"label" : label,
                            "n_pixels" : n_pixels,
                            "T": T,
                            "L": L,
                            "B": B,
                            "R": R}

                    # Adiciona o blob ?? lista
                    componentes.append(info)

                    # Atualiza o r??tulo
                    label = label + 1

    return componentes

def flood_fill(label, matrix, row, col):
    ''' Label: identificador do blob
        Matrix: a matriz que ser?? percorrida para encontrar e rotular os pixels
            que formam um blob
        Row: a coordenada Y do pixel
        Col: a coordenada X do pixel '''
        
    # Recebe o n??mero de linhas e colunas da imagem
    rows, cols, _ = matrix.shape

    # Altera o label do pixel
    matrix[row, col, 0] = label

    # Vari??veis que indicam as coordenadas do ret??ngulo que cont??m o blob
    T1 = row
    L1 = col
    B1 = row
    R1 = col

    # Vari??vel que recebe o n??mero total de pixels do blob
    # Inicializada com 1 para representar o pixel que est?? sendo visitado
    n_pixels = 1

    # Visita os pixels adjascentes ao pixel atual em vizinhan??a-4
    # Verifica se est??o dentro da imagem e se seu valor ?? 0, 
    # o que indica que um pixel faz parte de um blob mas n??o foi visitado
    # Os pixels s??o visitados recursivamente
    if row - 1 >= 0 and matrix[row - 1, col, 0] == 0:
        # Visita o pixel acima do pixel atual
        n_pixels_aux, T2, L2, B2, R2 = flood_fill(label, matrix, row - 1, col)
        # Soma o n??mero de pixels atual com o valor recebido do pixel adjascente
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
        
    # Retorna o n??mero de pixels do blob e as coordenadas que formam
    # o ret??ngulo que cont??m o blob 
    return n_pixels, T1, L1, B1, R1

def compare_pixels(T1, L1, B1, R1, T2, L2, B2, R2):
    ''' Compara dois conjuntos de coordenadas e retorna os valores das 
        coordenadas que est??o mais pr??ximas das bordas da imagem
        Esta fun????o ?? executada para encontrar as coordenadas mais extremas 
        do blob, permitindo que o ret??ngulo que cont??m o blob seja desenhado '''

    T = min(T1, T2)
    L = min(L1, L2)
    B = max(B1, B2)
    R = max(R1, R2)

    return T, L, B, R




if __name__ == "__main__":
    main()