# -*- coding: utf-8 -*-

#Roll No I170007 , I170025 , I170232
#Hassam Rawan Aman

# Import the cv2 library
import cv2 
import numpy as np
import cv2 as cv
from rlsa_smearing import rlsa
import imutils
from sklearn.ensemble import RandomForestClassifier

import  classifyingFunctions as cF
# Read the image you want connected components of
def show_img(img):
    cv2.imwrite('sds.jpg', img)
    img = cv2.imread('sds.jpg')
    cv2.imshow('sds', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sortContoursOrder(im_smeared,colorImage,contours,hier):
        # sorting code 
    img = im_smeared.copy()
    
    h,w = img.shape[:2]
    # sum all rows
    ret,thresh = cv.threshold(img,200,255,cv.THRESH_BINARY_INV)
    sumOfRows = np.sum(thresh, axis=1)
    # loop the summed values
    startindex = 0
    lines = []
    compVal = True
    for i, val in enumerate(sumOfRows):
        # logical test to detect change between 0 and > 0
        testVal = (val > 0)
        if testVal == compVal:
                # when the value changed to a 0, the previous rows
                # contained contours, so add start/end index to list
                if val == 0:
                    lines.append((startindex,i))
                # update startindex, invert logical test
                    startindex = i+1
                compVal = not compVal
    # create empty list
    lineContours = []
    # loop contours, find the boundingrect,
    # compare to line-values
    # store line number,  x value and contour index in list
    col2=colorImage.copy()
    for j,cnt in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(cnt)
        #cv2.rectangle(col2,(x,y),(x+w,y+h),(0,255,0),2)
        for i,line in enumerate(lines):
            if y >= line[0] and y <= line[1]:
                lineContours.append([line[0],x,j])
                break
    
    # sort list on line number,  x value and contour index
    #show_img(col2)
    contours_sorted = sorted(lineContours)
    return contours_sorted,col2

    
def RLSA_Smearing(image,value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_rlsa_horizontal_vertical = rlsa.rlsa(image_binary, True, True, value)
    #Apply both horizontal and vertical smearing.
    return image_rlsa_horizontal_vertical
# --------------------------------------------------------------------
def alphabetSegmentation(word,index):
    
    pred_sent=""
    img=word.copy()
    boxed_im=word.copy()
    words_img_list=[]
   
    # create black color image
    #blank_image= np.zeros((boxed_im.shape[0],boxed_im.shape[1],3), np.uint8)
    # change to grayscale two channel 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary_image = cv.threshold(gray,200,255,cv.THRESH_BINARY)
    
    # Apply vertical smearing to merge letters having dots such as i, j , k 
    # this will only check above and if there is dot it will become part of word 
    # this will help in detecting contours 
    image_rlsa_vertical = rlsa.rlsa(binary_image, False, True, 5)
    
    contours, hierarchy = cv.findContours(image_rlsa_vertical, cv2.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contours_img, hierarchy = cv.findContours(thresh, cv2.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_sorted,col2=sortContoursOrder(image_rlsa_vertical,boxed_im,contours,hierarchy)
    #show_img(image_rlsa_vertical)
    i=0
    unboxed_im=boxed_im.copy()
    for i,cnt in enumerate(contours_sorted):
        line, xpos, cnt_index = cnt
        if hierarchy[0][cnt_index][-1]==0:
            
            x,y,w,h = cv2.boundingRect(contours[cnt_index])
            pad=0
            x=x-pad
            y=y-pad
            w=w+(pad*2)
            h=h+(pad*2)
            
            cv2.rectangle(boxed_im,(x,y),(x+w,y+h),(0,255,0),1)
            #blank_image=cv.drawContours(blank_image, contours_img,i ,(255,255,255) ,-1 )

            crop_img = unboxed_im[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            ret,binary_image = cv.threshold(gray,200,255,cv.THRESH_BINARY)
            
            # do image translation in order to create padding for a extracted alphabet
#            num_rows, num_cols = binary_image.shape[:2]
#
#            translation_matrix = np.float32([ [1,0,10], [0,1,15] ])
#            img_translation = cv2.warpAffine(binary_image, translation_matrix, (num_cols + 30, num_rows + 30)) 
            #show_img(img_translation)
            dim = (60, 60)
            # resize image
            #resized = cv2.resize(binary_image, dim, interpolation = cv2.INTER_CUBIC)
            resized = imutils.resize(binary_image, width=60)
            cv2.imwrite('words/word'+str(index)+'_'+str(i)+'.jpg',resized)
            #all_imgs_alphabets.append(resized)
            words_img_list.append(resized)
               
        i+=1
    #show_img(boxed_im)  
    
    temp=cF.wordsTotext(words_img_list,Classifier)
    return temp          
        
#%%      




# train classifier 
Classifier=cF.runClassifier()

#%%

fileName='sample2.jpg'
predicted_sentence=""
smear_value=8
src = cv2.imread(fileName, 0)
col = cv2.imread(fileName, 1)
colorImage=col

#show_img(col)d

img = src.copy()
kernel = np.ones((5,5), np.uint8)
im_smeared=RLSA_Smearing(col.copy(),smear_value)
#show_img(im_smeared)

# find contours in order and crop them and then do alphabet segmentation 
ret,thresh = cv.threshold(im_smeared,200,255,cv.THRESH_BINARY)
show_img(thresh)
contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours_sorted = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
contours_sorted,col2=sortContoursOrder(im_smeared,colorImage,contours,hier)

words_cropped=[]
for i,cnt in enumerate(contours_sorted):
    line, xpos, cnt_index = cnt
    if hier[0][cnt_index][-1]==0:
        
        #cv2.putText(colorImage,str(i),(xpos,line+50),cv2.FONT_HERSHEY_SIMPLEX,1,(127),2,cv2.LINE_AA)
        (x,y,w,h) = cv2.boundingRect(contours[cnt_index])
        cv2.rectangle(col2,(x,y),(x+w,y+h),(0,255,0),2)
        pad=4
        x=x-pad
        y=y-pad
        w=w+(pad*2)
        h=h+(pad*2)
        crop_img = colorImage[y:y+h, x:x+w]
        #show_img(crop_img)
        #cv2.imwrite('words/word'+str(i)+'.jpg',crop_img)
        words_cropped.append(crop_img)    
show_img(col2)

for i in range(len(words_cropped)):
    
    p_sent=alphabetSegmentation(words_cropped[i],i)
    predicted_sentence+=p_sent 
    predicted_sentence+=" " 
print(predicted_sentence)
    # predict a word here 
    # add space and then predict next word 
    
    
    #space=cv2.imread('space.jpg')
    #cv2.imwrite('words/space'+str(i)+'.jpg',space)

cv2.destroyAllWindows()



            




