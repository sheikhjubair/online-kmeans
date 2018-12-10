# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:16:53 2018

@author: atifu
"""

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

cost_online = calculateCost(C, data)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=5)

tp=tn=fp=fn=0
for train_index, test_index in skf.split(data, labels):
    trData, testData = data[train_index,:], data[test_index, :]
    trLabel, testLabel = labels[train_index], labels[test_index]

    mdl = RandomForestClassifier(100, random_state=5)
    mdl.fit(trData, trLabel)
    predictedLabels = mdl.predict(testData)
    confMatrix = confusion_matrix(testLabel, predictedLabels)
    tp= tp  + confMatrix[0,0]
    tn= tn  + confMatrix[1,1]
    fp= fp  + confMatrix[2,2]
    



   




    