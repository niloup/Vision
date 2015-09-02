import sys
import numpy as np
import cv2
import cv
import math
import string
import cPickle as pickle
from time import time
import hog

import pdb

print "########Detection Using HOG-MEAN and SURF features#############"

SkipStep=8
BinNum=9
Angle=180
CellSize=8

num_pos=856
num_neg=778
num_train=num_neg+num_pos
num_test=92
print "#########End of setting Initial parameters########"
  
##trainMat contains all the HOG vectors for training images
trainMat=np.zeros((36,1))
##l_train contains the labels associated with each frame
l_train=np.zeros(num_train)
index_train=0

for i in range(num_pos):
    address="DataBase/training/pos/frame"+str(i+1)+".png"
    img=cv2.imread(address,0) 
    H=hog.ImgHOGFeature(img, SkipStep, BinNum, Angle, CellSize)
    if i==0:
       trainMat=H
    else:
       trainMat=np.vstack((trainMat,H))              
    l_train[index_train]=1
    index_train=index_train+1
print "#############End of Extracting HOG for Pos-Images############"

for i in range(num_neg):
    address="DataBase/training/neg/frame"+str(i+1)+".png"
    img=cv2.imread(address,0) 
    H=hog.ImgHOGFeature(img, SkipStep, BinNum, Angle, CellSize)   
    trainMat=np.vstack((trainMat,H))        
    l_train[index_train]=0
    index_train=index_train+1
print "#############End of Extracting HOG for Neg-Images############"

trainMat2=[]
l_train2=[]

for i in range(num_pos):
    address="DataBase/training/pos/frame"+str(i+1)+".png"
    train_image=cv.LoadImageM(address,cv.CV_LOAD_IMAGE_GRAYSCALE)
    storage=cv.CreateMemStorage() 
    k,d=cv.ExtractSURF(train_image,None,storage,(0,1000,3,4))
    del storage 
    
    trainMat2.append(d)           
    for i in range(len(k)):
        l_train2.append(1)   

for i in range(num_neg):
    address="DataBase/training/neg/frame"+str(i+1)+".png"
    train_image=cv.LoadImageM(address,cv.CV_LOAD_IMAGE_GRAYSCALE)
    storage=cv.CreateMemStorage() 
    k,d=cv.ExtractSURF(train_image,None,storage,(0,1000,3,4))
    del storage 
    if len(k)!=0:
       trainMat2.append(d)           
       for i in range(len(k)):
           l_train2.append(0)
         
trainMat2=np.vstack(trainMat2) 
l_train2=np.vstack(l_train2)
print "#########End of finding trainMat2 & l_train2############"

clf = cv2.SVM()
clf.train(trainMat.astype('float32'), l_train.astype('float32'))
print "######End of training the model using SVM on HOG features (trainMat)########"

clf2 = cv2.SVM()
clf2.train(trainMat2.astype('float32'), l_train2.astype('float32'))
print "######End of training the model using SVM on SURF features(trainMat2)########"

print "######Performing the sliding window of size <<128*64>> on each test data#######"
for pic_num in range(num_test):
    address="DataBase/testing/frame"+str(pic_num+1)+".png"
    image=cv2.imread(address,0)
    [height, width] = image.shape   
  
    ##xStepNum is the number of blocks I will have in the x direction
#    xStepNum = math.ceil(width*1./64)
    xStepNum = math.ceil((width-64)*1./32+1)
    ##yStepNum is the number of blocks I will have in the y direction
#    yStepNum = math.ceil(height*1./128)
    yStepNum = math.ceil((height-128)*1./64+1)
    
    result2=np.zeros((xStepNum,yStepNum))
        
    for i in range(int(xStepNum)):
        for j in range(int(yStepNum)):
            col = i*64
            row = j*128
                       
            block=np.zeros((128,64))
            
            if (row+128-1<=image.shape[0] and col+64-1<=image.shape[1]):
               block = image[row:row+128,col:col+64]
            
            elif (col+64-1>image.shape[1]):
                block[:,0:image.shape[1]-col] = image[row:row+128,col:image.shape[1]]
            
            elif (row+128-1>image.shape[0]):
               block[0:image.shape[0]-row,:] = image[row:image.shape[0],col:col+64]
                                                    
            H=hog.ImgHOGFeature(block, SkipStep, BinNum, Angle, CellSize)                        
            result2[i,j] = clf.predict(H.astype('float32')) 

            if (result2[i,j]==1):           
               storage=cv.CreateMemStorage() 
               k,d=cv.ExtractSURF(cv.fromarray(block),None,storage,(0,1000,3,4))
               del storage
               d=np.array(d)
               if (len(k)==0):
                   result2[i,j]=0               
               else:
                  x=np.zeros(len(k))
                  for g in range(len(k)):
                      x[g]=clf2.predict(d[g,:].astype('float32')) 
                      
                  if (np.sum(x)<10):
                     result2[i,j]=0 
                           
    print "#########End of associating all blocks in test image with a label############"
    
    index=np.where(result2==1)
    num1=index[1].shape[0]
    for t in range(num1):
        P_x=int(index[0][t])*64
        P_y=int(index[1][t])*128
        cv2.rectangle(image,(P_x,P_y),(P_x+64,P_y+128),(256.0,256.0,256.0),6)
    cv2.imwrite("output/frame-"+str(pic_num+1)+".png",image)
   
print "######### DONE!! You can see results now :-) ##########"    
    
