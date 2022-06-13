import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


userimage= input('Enter the image name:')
img1 = cv2.imread(userimage)



 

#canny edge detection
def canny(img1):
    return cv2.Canny(img1, 100, 200)

# Rotate the image around its center
def rotateImage(img1, angle: float):
    newImg =img1.copy()
    (h, w) = newImg.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImg = cv2.warpAffine(newImg, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #return newImage
    dst = cv2.fastNlMeansDenoisingColored(newImg,None, 10, 10, 7, 15)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    filter1 = cv2.medianBlur(blur,5)
    
    thresh =cv2.threshold(filter1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    white_pix = np.sum(thresh == 255)
    black_pix = np.sum(thresh == 0)
    if (white_pix<=black_pix):
        thresh =cv2.threshold(filter1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = np.ones((1,1),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations =5 )
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)

    #print('Number of white pixels:',white_pix)
    #print('Number of black pixels:',black_pix)
        
    plt.imshow(newImg,cmap=cm.gray)
    plt.axis('off')
    plt.show()

    plt.imshow(thresh,cmap=cm.gray)
    plt.axis('off')
    plt.show()
        
    cv2.imshow('erosion',erosion)
    cv2.waitKey(0)
    cv2.imwrite('final1.png',erosion)

    #cv2.imshow('dst',dst)
    #cv2.waitKey(0)
rotateImage(img1, 0)    
