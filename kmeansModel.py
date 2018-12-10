# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:03:25 2018

@author: Jubair
"""
import fileReading
import numpy as np

def getPereiraDataAndLabels():
    data = fileReading.readFile(r'data_RNA_Seq_expression_median_modified.txt')
    data = data.transpose()
    labels = fileReading.readFile(r'data_clinical_sample.txt')
    labels =  labels[4:,:]
    patientIDInLabels =labels[1:,0]
    patientIDInExpression = data[1:, 0]
    finalLabels = []
    for patientID in patientIDInLabels:
        ind = np.argwhere(patientIDInExpression == patientID)
        if ind.size  != 0:
            val = labels[ind[0],5].tolist()
            if len(val):
                finalLabels.append(val[0])
    
    data = data[1:,1:]
    data = np.array(data, dtype=float)
    
    nLabels= np.zeros((data.shape[0],), dtype = np.int)
    for idx in range(len(finalLabels)):
        if finalLabels[idx] == 'Positive':
            nLabels[idx] = 1
    return data, nLabels