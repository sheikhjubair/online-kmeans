# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:35:43 2018

@author: Jubair
"""

import numpy as np
import fileReading
from sklearn.cluster import KMeans

k = 5

data = fileReading.readFile('lung-cancer.data')
labels = data[:,0]
data = data[:,1:]
data = np.array(data) 
data = data.astype(float)

kmeans = KMeans(n_clusters= k, random_state=5).fit(data)

cost_offline = kmeans.inertia_
