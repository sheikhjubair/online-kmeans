# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:58:13 2018

@author: Jubair
"""
import csv
import numpy as np
def readFile(filename):
    data = []
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            data.append(row)

    data = np.array(data)
    
    
    return data

