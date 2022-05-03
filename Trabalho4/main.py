#===============================================================================
# Processamento Digital de Imagens
# Trabalho 4 - Segmentação
# Autor: Breno Moura de Abreu
# RA: 1561286
# Data: 2022-05-04
#===============================================================================

import cv2
import numpy as np

#=================================LEIA ME=======================================

'''
Ok, vamos lá...
Inicialmente preciso informar que o programa não funciona completamente 
nem corretamente. 
Acredito que a idea está correta, mas não consegui escrever 
um programa funcional, infelizmente.
Admito que não investi tanto tempo quanto devia com esse projeto. Me ocupei
com outros trabalhos e acabei não tendo tempo o suficiente para terminar este 
aqui nem procurar conselhos com o professor, o que teria sido muito útil, é claro.
Depois de pesquisar um pouco acabei encontrando uma solução envolvendo 
encontrar bordas côncavas na imagem.
Não cheguei a ler o artigo que apresenta os métodos usados, 
apenas li a introdução o que já me deu a base para utilizar essa ideia 
para encontrar a resposta.
Tentei encontrar as soluções por conta própria, mas não consegui criar um programa funcional.
O programa não realiza o threshold adaptativo, então só "funciona" para imagens
com um foreground e background com valores parecidos entre si

A ideia aqui é a seguinte:
Encontrar as bordas dos elementos (grãos de arroz) utilizando o gradiente
Consigo encontrar os elementos isolados apenas utilizando um flood fill
Os elementos que fazem parte de um conjunto de elementos interligados
constituem um único blob, e suas bordas são compartilhadas.
É possível notar que para os elementos conjuntos há bordas côncavas, duas para
cada elemento adicional. Ou seja, se dois grãos de arroz estão interligados,
haverão duas bordas cônvadas, se há três grãos interligados, haverão quatro
bordas côncavas, etc.
Utilizando essa lógica é apenas necessário achar uma forma de encontrar
e contabilizar essas bordas côncavas.
Para tal, tentei encontrar um ponto médio entre dois pontos distanciados
entre um valor constante de pixels, e verificar se ele pertence ao grão de arroz
ou ao background. Se faz parte do último, é uma borda côncava.
'''

#===============================================================================

def main():

    # Carrega a imagem
    img = cv2.imread("114.bmp")
    
    # Normaliza os valores da imagem para valores entre 0 e 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape((img.shape[0], img.shape[1], 1)) 
    img = img.astype(np.float32) / 255  
    img = cv2.GaussianBlur(img, (5, 5), 0)

    rows, cols = img.shape 

    img_bin = binarize(img)
    img_mag, dx, dy = find_edges(img_bin)
    img_ori = find_orientation(dx, dy)
    img_eroded = erode(img_mag)

    '''hsv = np.zeros_like(img_col)
    hsv[..., 0] = orientation
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(eroded, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("test_ori.png", bgr)'''

    img_eroded_copy = img_eroded.copy()
    elements, ex, ey = find_elements(rows, cols, img_ori, img_eroded_copy)
    count = count_elements(elements, ex, ey, img_bin, img_eroded)

    print("Total de elementos: " + str(count))

#===============================================================================

def count_elements(elements, ex, ey, img_bin, img_eroded):
    ''' Conta a quantidade total de elementos presentes na imagem
        Isso é feito da seguinte forma:
        Encontra-se o número de bordas côncavas total dos elementos
        Se não há uma borda côncava, o elmento está isolado
        Para cada conjunto de elementos aglomerados há duas bordas côncavas
        para cada elemento interligado. Dessa forma, divide-se por 2 o número
        de bordas côncavas presentes na imagem e soma-se ao número total
        de blobs para encontrar a quantidade de elementos únicos na imagem '''

    count = 0
    wnd_size = 12
    for i in range(len(elements)):
        arr = elements[i]
        arrx = ex[i]
        arry = ey[i]
        j = 0
        while j + wnd_size < len(arr):
            # Encontra a diferença entre a orientação de dois pontos 
            # Os pontos estão distanciados wnd_size pixels
            dif = abs(arr[j + wnd_size] - arr[j])
            if dif > 90:
                dif = abs(180 - dif)
            
            # Encontra o ponto médio entre os dois pixels
            midx = int(0.5 * arrx[j + wnd_size] + 0.5 *  arrx[j])
            midy = int(0.5 * arry[j + wnd_size] + 0.5 *  arry[j])

            # Verifica se o ponto médio faz parte do background ou não
            # Se sim, a borda é côncava
            if (img_bin[midy, midx] == 0 and img_eroded[midy, midx] == 0 and dif > 30):
                count = count + 1
                # Pula para o pixel da extermidade oposta e continua o algoritmo
                # a partir daquele ponto, já que a borda côncava foi encontrada.
                j = j + wnd_size
            else:
                j = j + 1
    
    # Retorna a quantidade total de elementos presentes na imagem
    return int(count / 2) + len(elements)

#===============================================================================

def find_elements(rows, cols, img_ori, img_eroded):
    ''' Encontra todos os elementos e armazena as coordenadas e a orientação
        de cada pixel que pertence ao elemento
        Retorna uma lista contendo todos os elementos '''
    elements = []
    array = []
    ex = []
    ey = []
    for y in range(rows):
        for x in range(cols):
            # Caso o pixel pertence a uma borda
            if img_eroded[y, x] != 0:
                array = []
                arrx = []
                arry = []
                # Realiza o flood fill para encontrar todos os pixels
                # que pertencem ao elemento, e armazena informações
                flood_fill(y, x, array, img_eroded, arrx, arry, img_ori)
                elements.append(array)
                ex.append(arrx)
                ey.append(arry)
    
    return elements, ex, ey

#===============================================================================

def find_orientation(dx, dy):
    ''' Encontra a orientação de cada pixel que pertence à uma borda '''
    orientation = cv2.phase(dx, dy, angleInDegrees=True)
    orientation = orientation / 2
    return orientation

#===============================================================================

def erode(img):
    ''' Realiza a erosão da imagem que apresenta as bordas dos elementos
        Isso é feito para afinar as bordas, permitindo trabalhar com menos pixels '''
    kernel = np.ones((2, 2))
    eroded = cv2.erode(img, kernel, iterations=1)
    return eroded

#===============================================================================

def find_edges(img):
    ''' Aplica o gradiente como filtro de Sobel para encontrar as bordas
        dos elementos, representado pela magnetude.'''
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    return mag, dx, dy

#===============================================================================

def binarize(img):
    ''' Binariza a imagem '''
    _, img_bin = cv2.threshold(img, 0.8, 1, cv2.THRESH_BINARY)
    return img_bin

#===============================================================================

def flood_fill(y, x, array, matrix, arrx, arry, orientation):
    ''' Realiza um flood fill nos contornos e armazena sua orientação
        Preenche arrays contendo as coordenadas e a orientação dos pixels '''
    rows, cols = matrix.shape
    array.append(orientation[y, x])
    arrx.append(x)
    arry.append(y)
    matrix[y, x] = 0

    if(y - 1 >= 0 and matrix[y - 1, x] != 0):
        flood_fill(y - 1, x, array, matrix, arrx, arry, orientation)
    if(x + 1 < cols and matrix[y, x + 1] != 0):
        flood_fill(y, x + 1, array, matrix, arrx, arry, orientation)
    if(y + 1 < rows and matrix[y + 1, x] != 0):
        flood_fill(y + 1, x, array, matrix, arrx, arry, orientation)
    if(x - 1 >= 0 and matrix[y, x - 1] != 0):
        flood_fill(y, x - 1, array, matrix, arrx, arry, orientation)

#===============================================================================

if __name__ == "__main__":
    main()