# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:35:43 2018

@author: Jubair
"""

from sklearn.cluster import KMeans
import kmeansModel
k = 31

data, labels = kmeansModel.getPereiraDataAndLabels()

kmeans = KMeans(n_clusters= k, random_state=5).fit(data)

cost_offline = kmeans.inertia_
