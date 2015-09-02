import sys
import numpy as np
import cv2
import cv
import math
import HOGGradient
import BinHOGFeature

#####Defining constants#####  
pi=3.14159265

def ImgHOGFeature(img, SkipStep, BinNum, Angle, CellSize):
        
    #####Gamma/Colour Normalization#####
    if (np.ndim(img)==3):
       G=cv2.cvtColor(img,cv.CV_BGR2GRAY)
    else:
       G=img
    [height, width] = G.shape
    
    #####Gradient and Gradient angle Computation#####
    [GradientX, GradientY]=HOGGradient.HOGGradient(G[:,:].astype('double'))
   
    ##calculate the norm of gradient##
    Gr=np.sqrt(np.square(GradientX)+np.square(GradientY))
    
    ##Calculate the angle##
    index=np.where(GradientX==0)
    GradientX[index]=1e-5
    
    if (Angle==180):
       A=((np.arctan(GradientY/GradientX)+(pi/2))*180)/pi
    if (Angle==360):
       A=((np.arctan2(GradientY,GradientX)+pi)*180)/pi

    ######Spatial / Orientation Binning#####
    nAngle = Angle/BinNum
    IndTag = np.ceil(A/nAngle)

    ##xStepNum is the number of blocks I will have in the x direction
    xStepNum = math.floor((width-2*CellSize)/SkipStep+1)
    ##yStepNum is the number of blocks I will have in the y direction
    yStepNum = math.floor((height-2*CellSize)/SkipStep+1)
    ##overL is the amount the block will be shifted forward: selected that to be 1./4*16 where 16 is the block size.
    overL = SkipStep   
    FeatDim = BinNum*4
    H = np.zeros((FeatDim, xStepNum*yStepNum))
    currFeat = np.zeros((FeatDim,1))

    for i in range(int(xStepNum)):
        for j in range(int(yStepNum)):
            x_Off = i*overL
            y_Off = j*overL
            ##Here we define each block have 4 cells##
            blockGr = Gr[y_Off:y_Off+2*CellSize-1,x_Off:x_Off+2*CellSize-1]
            blockInd = IndTag[y_Off:y_Off+2*CellSize-1,x_Off:x_Off+2*CellSize-1]
            ##calculate the block tag and Grain of each pixel##
            currFeat = BinHOGFeature.BinHOGFeature(blockGr, blockInd, CellSize, BinNum)       
            ##calculate the feature of the block##
            H[:,i*yStepNum+j] = currFeat
                               
    H=np.mean(H,axis=1)   

    return(H)
 

 
    



