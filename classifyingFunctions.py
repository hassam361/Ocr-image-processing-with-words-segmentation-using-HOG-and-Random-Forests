# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:27:26 2019
Roll No I170007 , I170025 , I170232

"""
#----------------------------------------------------------------------------------------------------------
#
#                   CLASSIFICATION PART
#
#------------------------------------------------------------------------------------------------------------

import os

import cv2
import cv2 as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
taxonomy="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,;:?!.@#$%&(){}[]" 


def train_classifier(Classifier):
    dataset_path='C:\\Users\\hassa\\Desktop\\dip project\\trainingData'
    
    train_samples=os.listdir(dataset_path)
  
    
    
    print(train_samples)
    
    paths_dic={}
    i=0
    dirs_list=[]
    for i in range(len(train_samples)):
        dirs_list.append(os.path.join(dataset_path,train_samples[i]))
    
    for i,idir in enumerate(dirs_list): 
        file_path_pngs=[]
        files=os.listdir(idir)
        for file in files:
            file_path_pngs.append(os.path.join(idir,file))
        paths_dic[train_samples[i]]=file_path_pngs
    imgs_dic={}
    
    sample_dirs=list(paths_dic.values())
    
    train_data=[]
    for i in range(len(paths_dic)):
        img_list=[]
        paths_in_sample=sample_dirs[i]
        for png_path in paths_in_sample:
            img=cv2.imread(png_path,0)
            img_list.append([train_samples[i],img])
            train_data.append([train_samples[i],img])
        imgs_dic[i]=img_list
        

    sample_img=train_data[0][1]
    sample_img=np.array(sample_img)
    width_data=sample_img.shape[0]
    height_data=sample_img.shape[1]
    
    train_num_labels=[]
    for line in train_data: 
        train_num_labels.append(line[0])
    
    X=[]
    for i in range(len(train_data)):
        image=train_data[i][1]
        dim = (width_data, height_data)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
        winSize = (32,32)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        h = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        hist = h.compute(resized)
    
        X.append(hist)
    X=np.array(X)
    X=np.squeeze(X)
    print ("X shape: ",X.shape)
    # Train dataset on Random Forest Classifier 
    
   
    Classifier.fit(X,train_num_labels)
    return Classifier

    
     
def wordsTotext(word_img,Classifier):
    predicted_words=""
    Xtest=[]
    for i in range(len(word_img)):
        
        hog_img=word_img[i]
        dim = (60, 60)
       
        # resize image
        resized = cv2.resize(hog_img, dim, interpolation = cv2.INTER_CUBIC)
        #show_img(resized)
        winSize = (32,32)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        h = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        hist = h.compute(resized)
        Xtest.append(hist)
    Xtest=np.array(Xtest)


    print(Xtest.shape)
    for i in range(len(word_img)):
        oneAlpha=Xtest[i].reshape(1,-1)
        image=word_img[i]
        
        label=Classifier.predict(oneAlpha)
        pred_alpha=taxonomy[int(label[0])-1]
        print(pred_alpha)
        predicted_words+=pred_alpha
    return predicted_words
def runClassifier():
    Classifier = RandomForestClassifier(n_estimators = 500, max_depth=None,
    min_samples_split=2, random_state=64,n_jobs=3,verbose=True)
    Classifier=train_classifier(Classifier)
    return Classifier