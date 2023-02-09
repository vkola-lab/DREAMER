# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2023

@author: Meysam Ahangaran
"""

import numpy as np
import numba


class ClassOverlap:
    
    k = 7   #kNN parameter
    tetha = 3   #threshold of R-Value
 
    def __init__(self):
        self
    
    
    #get number of samples of each five classes of DX_bl
    def getNumofSamples(self, data, className):
        rows = len(data)
        cols = len(data[0])
        ctr = 0
        for i in range(rows):
            if data[i][cols-1] == className:
                ctr = ctr + 1
        
        return ctr
    
    #get num of kNN for a class in the other class
    def getRvalueClass(self, data, className):
        
        rows = len(data)
        cols = len(data[0])
        data_num = np.array(data[0:(rows-1),0:(cols-1)], dtype=np.float32)
        data_class = np.array(data[:,(cols-1)], dtype=np.str0)
        
        kNN_ctr = 0 #number of samples in the className with overlapping property
        
        for i in range(rows):
            if data[i][cols-1] == className:
               kNN_otherClass = self.getKnnOtherClass(data_num, data_class, className, i)
               if kNN_otherClass > self.tetha:
                   kNN_ctr = kNN_ctr + 1
        
        return kNN_ctr
        
                
    #get num of Knn of a idx in other class of className
    def getKnnOtherClass(self, data_num, data_class, className, idx):
        
        kNN_otherClass = 0
        kNN_arrIndexes = _getKnnOtherClass(data_num, idx, self.k)
        for i in range(self.k):
            # print(data_class[kNN_arrIndexes[i]], className)
            if data_class[kNN_arrIndexes[i]] != className:
                kNN_otherClass = kNN_otherClass + 1
        
        return kNN_otherClass
    
    # calculate the R-Value for the input data (last column is DX_bl)
    def getRvalue(self, data):
        
        #five classes of DX_bl
        classCN = 'CN'
        classLMCI = 'LMCI'
        classEMCI = 'EMCI'
        classSMC = 'SMC'
        classAD = 'AD'
        
        CN_samples = self.getNumofSamples(data, classCN)
        LMCI_samples = self.getNumofSamples(data, classLMCI)
        EMCI_samples = self.getNumofSamples(data, classEMCI)
        SMC_samples = self.getNumofSamples(data, classSMC)
        AD_samples = self.getNumofSamples(data, classAD)
        
        totalSamples = CN_samples+LMCI_samples+EMCI_samples+SMC_samples+AD_samples

        
        kNN_CN = self.getRvalueClass(data, classCN)
        kNN_LMCI = self.getRvalueClass(data, classLMCI)
        kNN_EMCI = self.getRvalueClass(data, classEMCI)
        kNN_SMC = self.getRvalueClass(data, classSMC)
        kNN_AD = self.getRvalueClass(data, classAD)
        
        R_value = (kNN_CN+kNN_LMCI+kNN_EMCI+kNN_SMC+kNN_AD) / totalSamples
        # print('R value is ',R_value)
        return R_value

@numba.jit(nopython=True)
def _getKnnOtherClass(data, idx, k):
    
    rows = len(data)
    cols = len(data[0])
    distArr = np.zeros(rows, dtype=np.float32) #distances of samples to the idx sample
    kNN_arrIndexes = np.zeros(k, dtype=np.int32) #array of kNN indexess
    
    for i in range(rows):
        # distArr[i] = distance.euclidean(data[i, 0:(cols-1)], data[idx, 0:(cols-1)])
        v1 = data[i, :]
        v2 = data[idx, :]
        distArr[i] = np.sqrt(np.sum((v1 - v2) ** 2))
        
    for i in range(k):
        if idx!=0:
            minIdx = 0 #first element is the min element    
        else:
            minIdx = 1
        for j in range(rows):
            if (j==idx or j in kNN_arrIndexes):  #current index element is discarded
                continue
            if distArr[j] <= distArr[minIdx]:
                minIdx = j
        kNN_arrIndexes[i] = minIdx                                                           
    return kNN_arrIndexes

    