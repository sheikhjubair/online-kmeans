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
from sklearn.feature_selection import chi2
from ensembledRandomForest import classifyWithRandomForest, predictEnsembledRandomForest


k = 20

data, labels = kmeansModel.getPereiraDataAndLabels()
C= onlineKmeans(data, k)
kmeansFeatures = onlineKmeansFeature(C, data)


kmeansFeatures = np.array(kmeansFeatures)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
tp = 0
tn = 0
fn = 0
fp = 0
print('Calculating Cost')
cost_online = calculateCost(C, data)
print('Cost: {}'.format(cost_online))
for train_index, test_index in skf.split(kmeansFeatures, labels):
    trData, testData = kmeansFeatures[train_index,:], kmeansFeatures[test_index, :]
    trLabel, testLabel = labels[train_index], labels[test_index]
    
    mdls = classifyWithRandomForest(trData, trLabel,100)
    predicted = predictEnsembledRandomForest(mdls, testData)
    
    conf = confusion_matrix(testLabel, predicted)
    tp = tp + conf[0,0]
    tn = tn + conf[1,1]
    fn = fn + conf[0,1]
    fp = fp + conf[1,0]
    
print('Accuracy: {}'.format((tp + tn)/ (tp + tn + fn + fp)))
    

    
    



    