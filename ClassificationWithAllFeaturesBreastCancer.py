# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:18:15 2018

@author: Jubair
"""
import kmeansModel
import numpy as np
from onlineKmeans import onlineKmeans, onlineKmeansFeature
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from ensembledRandomForest import classifyWithRandomForest, predictEnsembledRandomForest


#data, labels = kmeansModel.getPereiraDataAndLabels()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 5)
tp = 0
tn = 0
fn = 0
fp = 0
i = 1
for train_index, test_index in skf.split(data, labels):
    print('Working with {}'.format(i))
    i+=1
    trData, testData = data[train_index,:], data[test_index, :]
    trLabel, testLabel = labels[train_index], labels[test_index]
    
    mdls = classifyWithRandomForest(trData, trLabel,100)
    predicted = predictEnsembledRandomForest(mdls, testData)
    
    conf = confusion_matrix(testLabel, predicted)
    tp = tp + conf[0,0]
    tn = tn + conf[1,1]
    fn = fn + conf[0,1]
    fp = fp + conf[1,0]
