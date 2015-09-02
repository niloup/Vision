import sys
import numpy as np
import cv2
import cv
import math

eps=0.01

def BinHOGFeature(blockGr, blockInd, CellSize, BinNum):

##Devide the block##
    block_ori1=blockGr[0:CellSize,0:CellSize]
    block_ori2=blockGr[0:CellSize,CellSize:2*CellSize]
    block_ori3=blockGr[CellSize:2*CellSize,0:CellSize]
    block_ori4=blockGr[CellSize:2*CellSize,CellSize:2*CellSize]
    
    block_grad1=blockInd[0:CellSize,0:CellSize]
    block_grad2=blockInd[0:CellSize,CellSize:2*CellSize]
    block_grad3=blockInd[CellSize:2*CellSize,0:CellSize]
    block_grad4=blockInd[CellSize:2*CellSize,CellSize:2*CellSize]

##Here we calculate 4 cells##
    binfeat = np.zeros((BinNum*4, 1))
    feat1 = np.zeros((BinNum, 1))
    feat2 = np.zeros((BinNum, 1))
    feat3 = np.zeros((BinNum, 1))
    feat4 = np.zeros((BinNum, 1))

    for i in range(BinNum):
	feat1[i] = np.sum(block_ori1[np.where(block_grad1==i)])
   
    for i in range(BinNum):
	feat2[i] = np.sum(block_ori2[np.where(block_grad2==i)])
         
    for i in range(BinNum):
	feat3[i] = np.sum(block_ori3[np.where(block_grad3==i)])
	
    for i in range(BinNum):
        feat4[i] = np.sum(block_ori4[np.where(block_grad4==i)])

    tmp1 = np.vstack((feat1,feat2))
    tmp2 = np.vstack((feat3,feat4))
    binfeat=np.vstack((tmp1,tmp2))
    binfeat=np.array(binfeat)
    binfeat=np.squeeze(binfeat)

    binfeat = binfeat/np.sum(binfeat)    
    sump=np.sqrt(np.sum(np.square(binfeat)))
    binfeat = binfeat/(sump+eps)
    return(binfeat)


