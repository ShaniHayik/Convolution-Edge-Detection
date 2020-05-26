import math
from collections import defaultdict
from math import sqrt
import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter



def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    size = len(inSignal) + len(kernel1) - 1
    ArrayImg = np.copy(inSignal)
    ArrayImg = np.resize(ArrayImg,size)
    i = size - len(inSignal)
    while (i<size):
        ArrayImg[i] = 0
        i=i+1

    arrayAns = []
    i = 0
    j=0
    sum = 0

    for j in range(size):
        for i in range(len(inSignal)-1):
            sum = sum + ArrayImg[j-i]*kernel1[i]

        arrayAns.append(sum)
        sum = 0

    return arrayAns


def transform2D(img):
    img_copy = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_copy[i,j] = img[img.shape[0]-i-1][img.shape[1]-j-1]
    return img_copy


def transform1D(img):
    img_copy = img.copy()
    for i in range(img.shape[0]):
        for j in range(1):
            img_copy[i] = img[img.shape[0]-i-1]
    return img_copy


def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    img_h = inImage.shape[0]
    img_w = inImage.shape[1]

    if (np.size(np.shape(kernel2)) == 1):
        kernel = transform1D(kernel2)
        kernel_h = kernel.shape[0]
        kernel_w = 1

        h = kernel_h // 2
        w = kernel_w // 2

        img_conv = np.zeros(inImage.shape)
        for i in range(h, img_h - h):
            for j in range(w, img_w - w):
                sum = 0
                for m in range(kernel_h):
                    for n in range(1):
                        sum = sum + kernel[m] + inImage[i - h + m][j - w + n]

                img_conv[i, j] = sum

    if (np.size(np.shape(kernel2)) == 2):
        kernel = transform2D(kernel2)
        kernel_h = kernel.shape[0]
        kernel_w = kernel.shape[1]

        h = kernel_h//2
        w = kernel_w//2

        img_conv = np.zeros(inImage.shape)
        for i in range(h, img_h-h):
            for j in range(w, img_w-w):
                sum = 0
                for m in range(kernel_h):
                    for n in range(kernel_w):
                        sum = sum + kernel[m][n] + inImage[i-h+m][j-w+n]

                img_conv[i,j] = sum

    return img_conv



def convDerivative(inImage:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    new_img = np.copy(inImage)
    kernel_x = np.array([-1,0,1])
    kernel_y = np.array([[-1],
                [0],
                [1]])

    der_x = cv2.filter2D(new_img, -1, kernel_x)
    der_y = cv2.filter2D(new_img, -1, kernel_y)


    magnitude = np.zeros(np.shape(new_img))
    direction = np.zeros(np.shape(new_img))
    h_img, w_img, size = np.shape(new_img)

    for i in range(h_img-1):
        for j in range(w_img-1):
            a = pow(der_x[i,j],2)
            b = pow(der_y[i,j],2)
            c = der_y[i, j] / der_x[i, j]

            magnitude[i,j] = sqrt(a[0]+b[0])
            direction[i,j] = math.degrees(math.atan(c[0]))


    return direction,magnitude,der_x,der_y




def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

    sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])*1/8
    sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])*1/8

    conv_x = cv2.filter2D(img,-1, sx)
    conv_y = cv2.filter2D(img, -1, sy)

    sobel_mine = np.zeros(np.shape(img))
    h_img, w_img, size = np.shape(img)

    for i in range(h_img-1):
            for j in range(w_img-1):
                a = pow(conv_x[i,j],2)
                b = pow(conv_y[i,j],2)
                sobel_mine[i,j] = sqrt(a[0]+b[0])

    return sobel, sobel_mine*5



def edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    ans = gaussian_filter(img, sigma=2)
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    ans = cv2.filter2D(ans, -1, kernel)
    # ans = conv2D(img,kernel)
    return ans



def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)-> (np.ndarray, np.ndarray):
    ans = gaussian_filter(img, sigma=3)

    ans2 = (ans * 255).astype(np.uint8)
    ans2 = cv2.Canny(ans2, thrs_1, thrs_2)

    ans = ans * 50
    sobelx = cv2.Sobel(ans, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(ans, cv2.CV_64F, 0, 1, ksize=5)

    a = np.zeros(np.shape(ans))
    b = np.zeros(np.shape(ans))
    c = np.zeros(np.shape(ans))
    magnitude = np.zeros(np.shape(ans))
    direction = np.zeros(np.shape(ans))
    h_img, w_img, size = np.shape(ans)

    for i in range(h_img - 1):
        for j in range(w_img - 1):
            a[i,j] = pow(sobelx[i, j], 2)
            b[i,j] = pow(sobely[i, j], 2)
            #c[i,j] = sobely[i, j] / sobelx[i, j]
            #print(sobelx[i, j])

            #magnitude[i, j] = sqrt(a[i,j,0] + b[i,j,0])
            direction[i, j] = np.arctan2(sobely[i, j], sobelx[i, j])

    a, magnitude,b,c = convDerivative(ans)
    magnitude = non_max_suppression(magnitude,direction)


    #res, weak, strong =threshold(magnitude, lowThresholdRatio=0.002, highThresholdRatio=0.009)
    ans = hysteresis(magnitude, thrs_1,thrs_2)

    return ans2,ans


def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col,t = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col,0]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction<= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col,0] >= before_pixel[0] and gradient_magnitude[row, col,0] >= after_pixel[0]:
                output[row, col] = gradient_magnitude[row, col]

    return output


def hysteresis(img, weak, strong):
    #print(img)
    M, N,t = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j,0] == weak):
                try:
                    if ((img[i+1, j-1,0] == strong) or (img[i+1, j,0] == strong) or (img[i+1, j+1,0] == strong)
                        or (img[i, j-1,0] == strong) or (img[i, j+1,0] == strong)
                        or (img[i-1, j-1,0] == strong) or (img[i-1, j,0] == strong) or (img[i-1, j+1,0] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N,t = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j,a = np.where(img >= highThreshold)
    #zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j,b = np.where((img <= highThreshold) & (img >= lowThreshold))
    #print(res[0])
    res[strong_i] = strong
    res[weak_i] = weak

    return (res, weak, strong)




def houghCircle(input,circles)->list:
    rows = np.shape(input)[0]
    cols = np.shape(input)[1]

    # initializing the angles to be computed
    sinang = dict()
    cosang = dict()

    # initializing the angles
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)

    length = int(rows / 2)
    radius = [i for i in range(5, length)]

    # Initial threshold value
    threshold = 190

    for r in radius:
        # Initializing an empty 2D array with zeroes
        acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

        # Iterating through the original image
        for x in range(rows):
            for y in range(cols):
                if input[x][y] == 255:  # edge
                    # increment in the accumulator cells
                    for angle in range(0, 360):
                        b = y - round(r * sinang[angle])
                        a = x - round(r * cosang[angle])
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            a = int(a)
                            b = int(b)
                            acc_cells[a][b] += 1

        #print('For radius: ', r)
        acc_cell_max = np.amax(acc_cells)
        #print('max acc value: ', acc_cell_max)

        if (acc_cell_max > 150):

            #print("Detecting the circles for radius: ", r)

            # Initial threshold
            acc_cells[acc_cells < 150] = 0

            # find the circles for this radius
            for i in range(rows):
                for j in range(cols):
                    if (i > 0 and j > 0 and i < rows - 1 and j < cols - 1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j] + acc_cells[i - 1][j] + acc_cells[i + 1][j] +
                                              acc_cells[i][j - 1] + acc_cells[i][j + 1] + acc_cells[i - 1][j - 1] +
                                              acc_cells[i - 1][j + 1] + acc_cells[i + 1][j - 1] + acc_cells[i + 1][
                                                  j + 1]) / 9)
                        #print("Intermediate avg_sum: ", avg_sum)
                        if (avg_sum >= 33):
                            #print("For radius: ", r, "average: ", avg_sum, "\n")
                            circles.append((i, j, r))
                            acc_cells[i:i + 5, j:j + 7] = 0

    return circles


