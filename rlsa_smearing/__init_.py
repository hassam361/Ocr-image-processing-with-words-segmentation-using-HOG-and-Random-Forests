import rlsa
value = 10
import cv2
import numpy as np
def show_img(img):
    cv2.imwrite('sds.jpg', img)
    img = cv2.imread('sds.jpg')
    cv2.imshow('sds', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
image = cv2.imread('phase1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)




def nothing(x):
    pass

# Create a black image, a window

cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('SmearConstant','image',0,255,nothing)


kernel = np.ones((5,5), np.uint8)
image_rlsa_horizontal=image_binary[:]
while(1):
    cv2.imshow('image',image_rlsa_horizontal)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
   
    
    iterate_smear = cv2.getTrackbarPos('SmearConstant','image')
    image_rlsa_horizontal = rlsa.rlsa(image_binary, True, False, iterate_smear)
    
   



cv2.destroyAllWindows()