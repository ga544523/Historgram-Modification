import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import pandas as pd
import csv
import gc
import torch.nn.functional as F
import pandas as pd
from random import shuffle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import cmath
import math
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import random
import threading
import heapq as hq
import time
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def solve(origine,anshis):
    div=512*512

    cdf=[]
    sum=0
    for i in range(0,256,1):
        sum+=anshis[i]/div
        cdf.append(sum)

    for i in range(0,512,1):
        for j in range(0,512,1):
            origine[i][j]=255*cdf[origine[i][j]]

    cv2.imshow('result', origine)
    cv2.waitKey()

    return


if __name__ == '__main__':
    img_input = cv2.imread('1840.png', 0)

    equ = cv2.equalizeHist(img_input)


    his=[0]*256
    equhis=[0]*256
    orihis=[0]*256
    for i in range(0,512,1):
        for j in range(0,512,1):
            orihis[img_input[i][j]]+=1
            his[img_input[i][j]]+=1
            equhis[equ[i][j]]+=1

    hisarray=np.array(his)
    hisarray=hisarray.astype(float)
    equhis=np.array(equhis)
    equhis=equhis.astype(float)

    orihis=np.array(orihis)
    orihis=orihis.astype(float)

    gamma=1000.0
    lamb=1.0
    D=np.zeros([255, 256],dtype=float)
    for i in range(0,255,1):
        D[i][i]=-1
    for i in range(0,255,1):
        D[i][i+1]=1
    
    unit=np.zeros(([256,256]),dtype=float)
    for i in range(0,256,1):
        unit[i][i]=1.0+lamb

    ans=np.matmul(D.transpose(),D)*gamma+unit
    inverse_ans = np.linalg.inv(ans)

    ans_his=np.matmul(inverse_ans,hisarray+(lamb*equhis))
    print(equhis)
    print(ans_his)

    test=1
    if(test==0):
        plt.subplot(131)
        plt.plot(orihis)
        plt.subplot(132)
        plt.plot(equhis)
        plt.subplot(133)
        plt.plot(ans_his)
        plt.show()
    else:
        cv2.imshow('ori', img_input)
        cv2.imshow('equal', equ)
        solve(img_input,ans_his)

