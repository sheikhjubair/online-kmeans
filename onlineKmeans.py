# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:38:32 2018

@author: Jubair
"""
import numpy as np
import scipy.spatial

def onlineKmeans(V, K):
    k = np.ceil((K - 15)/5)
    
    C = V[:K + 10, :]
    
    w_star = calculate_WStar(C)
    print(w_star)
    
    r = 1
    q_r = 0
    
    f_r = w_star
    i =0
    for v in V[10:,:]:
        
        dist = calculateMinDistance(v, C)**2
        v = np.reshape(v, (1, v.shape[0]))
        if dist > f_r:
            C = np.append(C, v, axis = 0)
            q_r = q_r + 1
        elif dist / f_r > 1:
            C = np.append(C, v, axis = 0)
            q_r = q_r + 1
            print(C.shape[0])
        
        
        if q_r >= k:
            r = r + 1
            q_r = 0
            f_r = 10 * f_r
            
    return C

def onlineKmeansFeature(centers, data):
#    kmeansFeatures = []
#    for _d in data:
#        distances = calculateEuclidean(centers, _d)
#        kmeansFeatures.append(distances)
    
    kmeansFeatures = np.zeros((data.shape[0], centers.shape[0]))
    i = 0
    for _d in data:
        distanceInd = findIndMinDistance(_d, centers)
        kmeansFeatures[i, distanceInd] = 1
        i+=1
    return kmeansFeatures
              

def calculateEuclidean(v1, v2):
    v = (v1- v2)**2
    v = np.sum(v, axis = 1)
    v =  np.sqrt(v)
    
    return v

def calculate_WStar(V):
    indices = np.arange(0, V.shape[0])
    distMatrix = np.zeros((V.shape[0], V.shape[0]))
    for v in range(V.shape[0]):
        curIndices = np.delete(indices, v)
        _x = calculateEuclidean(V[curIndices, :], V[v,:])
        distMatrix[v,v] = 0
        
        distMatrix[v,0: v] = _x[0:v] 
        distMatrix[v, v+1: V.shape[0]] = _x[v:]
    
    upperTrianIndices = np.triu_indices(V.shape[0])
    distMatrix = distMatrix[upperTrianIndices]
    nonZeroIndex = np.nonzero(distMatrix)
    distMatrix = distMatrix[nonZeroIndex]
    distMatrix = np.sort(distMatrix)
    distMatrix = distMatrix[0:10]
    w_star = np.sum(distMatrix**2)/2
    return w_star

def calculateMinDistance(v, C):
    x = calculateEuclidean(C, v)
    minDistance = np.min(x)

    return minDistance

def findIndMinDistance(v, C):
    x = calculateEuclidean(C, v)
    minDistIndex = np.argmin(x)
    return minDistIndex

def calculateCost(centers, data):
    costs = np.zeros((centers.shape[0],))
    
    for _d in data:
        minCost = calculateMinDistance(_d, centers)
        ind = findIndMinDistance(_d, centers)
        costs[ind] += minCost**2
        
    return np.sum(costs)
        
    

#data = np.random.randint(1, 100, size=(500,20))

        
        
        