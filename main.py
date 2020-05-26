from ex2_utils import *
import matplotlib.pyplot as plt



def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    im = cv2.imread(filename, representation-1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/np.max(im)
    return im


def main():
    print("ID: 209372937")


    #**************CONV1D******************

    kernel = [0,-1,1,0,0]
    array = [0,0,1,0,0]
    ans = conv1D(array,kernel)
    ans2 = np.convolve(array, kernel, 'full')
    print("our sulotion for conv1D:", ans)
    print("np sulotion:", ans2)


    #**************CONV2D******************

    img_path = 'cat.jpg'
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])*1/9
    img = imReadAndConvert(img_path,1)
    borderType=cv2.BORDER_REPLICATE

    # out_them = cv2.filter2D(img, -1, kernel)
    # plt.imshow(out_them )
    # plt.show()
    # out_them = cv2.filter2D(img, -1, kernel, borderType)
    # plt.imshow((out_them * 255).astype(np.uint8))
    # plt.show()
    print("CONV2D")
    out_us = conv2D(img,kernel)
    plt.imshow((out_us*20).astype(np.uint8))
    plt.show()



    #**************CONV-Derivative******************


    img_path = 'pic2.jpeg'
    img = imReadAndConvert(img_path,1)
    direction, magnitude,der_x,der_y = convDerivative(img)
    print("Conv Derivative")
    plt.imshow((magnitude* 255).astype(np.uint8))
    plt.show()
    plt.imshow(der_y*5)
    plt.show()
    plt.imshow(der_x*5)
    plt.show()


    # **************edgeDetectionSobel******************

    img_path = 'new3.jpg'
    img = imReadAndConvert(img_path,1)
    sobel, Mysobel = edgeDetectionSobel(img)
    # plt.imshow(sobel*5)
    # plt.show()
    print("Edge Detection Sobel")
    plt.imshow((Mysobel * 255).astype(np.uint8))
    plt.show()


    # **************edgeDetectionZeroCrossingSimple******************

    img_path = 'cln1.png'
    img = imReadAndConvert(img_path, 1)
    ans = edgeDetectionZeroCrossingSimple(img)
    print("Edge Detection Zero Crossing Simple")
    plt.imshow((ans * 255).astype(np.uint8))
    plt.show()

    # **************edgeDetectionCanny******************

    img_path = 'lenagrey.png'
    img = imReadAndConvert(img_path, 1)
    ans2,ans1 = edgeDetectionCanny(img, 0.3,0.7)
    print("Edge Detection Canny")
    plt.imshow(ans1)
    plt.show()

    # **************houghCircle******************

    img_path = 'coins2.png'
    img = imReadAndConvert(img_path, 1)
    smoothed_img = gaussian_filter(img, sigma=2)
    smoothed_img = (smoothed_img * 255).astype(np.uint8)
    edged_image = cv2.Canny(smoothed_img, 100, 200)
    circles = []
    houghCircle(edged_image, circles)
    for vertex in circles:
        cv2.circle(img, (vertex[1], vertex[0]), vertex[2], (0, 255, 0), 1)
        cv2.rectangle(img, (vertex[1] - 2, vertex[0] - 2), (vertex[1] - 2, vertex[0] - 2), (0, 0, 255), 3)

    print(circles)
    print("Hough Circle")

    cv2.imshow('Circle Detected Image', img)
    cv2.imwrite('Circle_Detected_Image.jpg', img)
    img_path = 'Circle_Detected_Image.jpg'
    img = imReadAndConvert(img_path, 2)
    plt.imshow((img * 255).astype(np.uint8))
    plt.show()










if __name__ == '__main__':
    main()