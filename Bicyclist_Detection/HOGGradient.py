import sys
import numpy as np
import cv2
import cv
import math

def HOGGradient(G):

##Defining the mask as [-1, 0, 1]##
##The edge of the image uses [-1, 1]##
    height=G.shape[0]
    width=G.shape[1]

    gradx = np.zeros((height, width))
    grady = np.zeros((height, width))

##Process the Image edge##
    gradx[:,0] = np.sum(G[:,0:2]*np.tile([-1, 1], (height, 1)), axis=1)
    gradx[:,width-1] = np.sum(G[:, width-1:width]*np.tile([-1, 1], (height, 1)), axis=1)

    grady[0,:] = np.sum(G[0:2, :]*np.tile([[-1], [1]],( 1, width)), axis=0);
    grady[height-1,:] = np.sum(G[height-2:height, :]*np.tile([[-1],[1]], (1, width)), axis=0)
    
    mask=np.array([-1, 0, 1])
   
    for i in range(1,height-1):
        for j in range(1,width-1):
            tmpx = G[i, j-1:j+2]
            tmpy = G[i-1:i+2, j]
            gradx[i,j] = np.sum(tmpx*mask, axis=0)
            grady[i,j] = np.sum(tmpy*mask, axis=0)
    
    return(gradx, grady)
    
