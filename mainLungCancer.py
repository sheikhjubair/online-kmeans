# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:17:29 2018

@author: Jubair
"""

import kmeansModel
import numpy as np
from onlineKmeans import onlineKmeans, onlineKmeansFeature, calculateCost
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import fileReading
from sklearn import svm


data = fileReading.readFile('lung-cancer.data')
labels = data[:,0]
data = data[:,1:]
data = np.array(data) 
data = data.astype(float)

C= onlineKmeans(data, 5)
kmeansFeatures = onlineKmeansFeature(C, data)
cost_online = calculateCost(C, data)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=5)
kmeansFeatures=np.array(kmeansFeatures)
tp=tn=tnp=fn=0
for train_index, test_index in skf.split(data, labels):
    trData, testData = data[train_index,:], data[test_index, :]
    trLabel, testLabel = labels[train_index], labels[test_index]

    mdl = RandomForestClassifier(100, random_state=5)
    mdl.fit(trData, trLabel)
    predictedLabels = mdl.predict(testData)
    confMatrix = confusion_matrix(testLabel, predictedLabels)
    tp= tp  + confMatrix[0,0]
    tn= tn  + confMatrix[1,1]
    tnp= tnp  + confMatrix[2,2]
    


   




    