# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:35:07 2018

@author: Jubair
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def classifyWithRandomForest(data, target, numIterations):
    
    
    uniqueLabels = np.unique(target)
    
    label1Count = np.sum(target == uniqueLabels[0])
    label2Count = np.sum(target == uniqueLabels[1])
    minorLabel = 0
    majorLabel = 0
    if label1Count < label2Count:
        ratio = label2Count / label1Count
        minorLabel = uniqueLabels[0]
        majorLabel = uniqueLabels[1]
    else:
        ratio = label1Count / label2Count
        minorLabel = uniqueLabels[1]
        majorLabel = uniqueLabels[0]
        
    if ratio>3:
        minorIndex = target == minorLabel
        minorData = data[minorIndex,:]
        minorLabels  = target[minorIndex]
        minorPopulation = minorData.shape[0]
        majorData = data[~minorIndex, :]
        majorLabels  = target[~minorIndex]
        mdls = []
        for _i in range(numIterations):
            np.random.seed(_i)
            majorTrIndex= np.arange(majorData.shape[0])
            np.random.shuffle(majorTrIndex)
            majorTrData= majorData[majorTrIndex[:minorPopulation], :]
            majorTrLabels = majorLabels[majorTrIndex[:minorPopulation]]
            mdl = RandomForestClassifier(100, random_state=5)
            
            trData = np.append(minorData, majorTrData, axis = 0)
            trLabels =  np.concatenate((minorLabels, majorTrLabels), axis = None)
            mdl.fit(trData, trLabels)
            
            mdls.append(mdl)
        
        return mdls
            
        
def predictEnsembledRandomForest(mdls, testData):
    
    allPredictions = np.zeros((testData.shape[0], len(mdls))) 
    for _i in range(len(mdls)):
        predicted = mdls[_i].predict(testData)
        allPredictions[:,_i] = predicted
        
    avgPredictions = np.average(allPredictions, axis = 1)
    avgPredictions= avgPredictions < 0.5
    
    avgPredictions = np.array(avgPredictions, dtype = np.int)
    
    return avgPredictions
    
        
        