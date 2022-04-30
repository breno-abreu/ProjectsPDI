import cv2
import numpy as np
from sqlalchemy import false

def main():
    # Carrega a imagem
    img = cv2.imread("150.bmp")
    img_col = cv2.imread("150.bmp")

    # Normaliza os valores da imagem para valores entre 0 e 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.reshape((img.shape [0], img.shape [1], 1))    
    img = img.astype(np.float32) / 255

    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("test_blurred.png", img * 255)

    row, col = img.shape

    ret, img = cv2.threshold(img, 0.8, 1, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cv2.imwrite("test_bin.png", img * 255)

    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(dx, dy)
    cv2.imwrite("test_mag.png", mag * 255)

    kernel = np.ones((2, 2))
    eroded = cv2.erode(mag, kernel, iterations=1)
    cv2.imwrite("test_eroded.png", eroded * 255)

    orientation = cv2.phase(dx, dy, angleInDegrees=True)
    orientation = orientation / 2

    hsv = np.zeros_like(img_col)
    hsv[..., 0] = orientation
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(eroded, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("test_ori.png", bgr)

    aux = eroded.copy()

    blobs = []
    bx = []
    by = []
    label = 1
    array = []
    aux2 = eroded.copy()
    for y in range(row):
        for x in range(col):
            if aux[y, x] != 0:
                array = []
                arrx = []
                arry = []
                flood_fill(y, x, array, arrx, arry, orientation, aux, label, aux2)
                label = label + 1
                blobs.append(array)
                bx.append(arrx)
                by.append(arry)
    
    cv2.imwrite("test_more_eroded.png", aux2 * 255)
    
    np.savetxt("foo.csv", blobs[2], delimiter=",")
    
    points = eroded.copy()

    count = 0
    plus = 12

    for i in range(len(blobs)):
        arr = blobs[i]
        arrx = bx[i]
        arry = by[i]
        j = 0
        while j + plus < len(arr):
            dif = abs(arr[j + plus] - arr[j])
            if dif > 90:
                dif = abs(180 - dif)
            
            #midx = int(min(arrx[j + plus], arrx[j]) + abs(arrx[j + plus] - arrx[j]) / 2)
            #midy = int(min(arry[j + plus], arry[j]) + abs(arry[j + plus] - arry[j]) / 2)
            midx1 = int(0.7 * arrx[j + plus] + 0.3 *  arrx[j])
            midy1 = int(0.7 * arry[j + plus] + 0.3 *  arry[j])
            midx2 = int(0.3 * arrx[j + plus] + 0.7 *  arrx[j])
            midy2 = int(0.3 * arry[j + plus] + 0.7 *  arry[j])
            if (img[midy1, midx1] == 0 and img[midy2, midx2] == 0 and 
                eroded[midy1, midx1] == 0 and eroded[midy2, midx2] == 0 and dif > 30):
                points[midy1, midx1] = 0.5
                points[midy2, midx2] = 0.5
                count = count + 1
                j = j + plus
            else:
                j = j + 1
    
    cv2.imwrite("test_eros.png", eroded * 255)
    cv2.imwrite("test_points.png", points * 255)

    print(int(count / 2) + len(blobs))

def flood_fill(y, x, array, arrx, arry, orientation, aux, label, aux2):
    rows, cols = aux.shape
    array.append(orientation[y, x])
    arrx.append(x)
    arry.append(y)
    aux[y, x] = 0
    count = 0

    if(y - 1 >= 0 and aux[y - 1, x] != 0):
        flood_fill(y - 1, x, array, arrx, arry, orientation, aux, label, aux2)
    if(x + 1 < cols and aux[y, x + 1] != 0):
        flood_fill(y, x + 1, array, arrx, arry, orientation, aux, label, aux2)
    if(y + 1 < rows and aux[y + 1, x] != 0):
        flood_fill(y + 1, x, array, arrx, arry, orientation, aux, label, aux2)
    if(x - 1 >= 0 and aux[y, x - 1] != 0):
        flood_fill(y, x - 1, array, arrx, arry, orientation, aux, label, aux2)


if __name__ == "__main__":
    main()