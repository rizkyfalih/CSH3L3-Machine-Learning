# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 08:28:21 2018

@author: Rizky Falih
"""

# Probabilistic Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as la


def g(x,sigma,w) :
    y = x.sub(w, axis = 0) #data subtraction
    y = la.norm(y.values)
    return np.exp(- y / (2*sigma)**2)

def patternLayer(data,dataTrain, sigma_dict):
    for i in dataTrain.index:
        this_label = int(dataTrain.loc[i,'label'])
        dataTrain.loc[i,'sigma'] = sigma_dict[this_label]
        dataTrain.loc[i,'g'] = g(data, sigma_dict[this_label], dataTrain.loc[i,'att1':'att3'])
    return dataTrain

def f(g, sigma):
    return g.sum()['g'] / ((2*np.pi)**(3/2) * sigma**3 * len(g.index) )

def f_sigma(data, a):
    idx = data.index
    d = []
    for i in idx:
        currentData  = data.loc[i]
        neighborData = data.drop(data.index[i]) 
        neighborData = neighborData.sub(currentData,axis=1).values #Distance matrix
        norms = [la.norm(j) for j in neighborData ]
        d.append(min(norms)) #adding the minimum distance value
    return (a * np.mean(d))

def summation(data, sigma):
    classes = np.unique(data['label']) #find how many distinct classes 
    f_values = {}
    for i in range (len(classes)):
        class_i = data.loc[data['label'] == classes[i]].reset_index(drop = True)
        sigma_i = sigma[classes[i]]
        f_values.update({f(class_i, sigma_i):classes[i]})
    return f_values

def outputLayer(f_values):
    maximum = -1
    for key in f_values :
        if maximum < key:
            maximum = key
    return f_values[maximum] 

def classify(data, dataTrain, a):
    
    #Todo: Finding the sigmas 
    sigmas = {}
    classlist = np.unique(dataTrain['label'])
    for i in range (len(classlist)):
        
        class_i = dataTrain.loc[dataTrain['label'] == classlist[i]].reset_index(drop = True)
        class_i = class_i.iloc[:,0:4]
        sigmas.update({classlist[i]:f_sigma(class_i, a)})
    
    #Pattern Layer
    plo = patternLayer(data,dataTrain, sigmas)
    
    #Summation Layer
    f_values = summation(plo, sigmas)
    #output Layer
    newLabel = outputLayer(f_values)
    return newLabel


# Get the dataset
dataTrain = pd.read_csv('data_train_PNN.txt', delimiter = "\t").round(9)
dataTest = pd.read_csv('data_test_PNN.txt', delimiter = "\t").round(9)

# Classify from the label
class0 = dataTrain.loc[dataTrain['label'] == 0].reset_index(drop = True)
class1 = dataTrain.loc[dataTrain['label'] == 1].reset_index(drop = True)
class2 = dataTrain.loc[dataTrain['label'] == 2].reset_index(drop = True)

a = 1 #assume that a = 1
newDataTrain = dataTrain.iloc[:,0:4]
outputData = dataTrain.iloc[1:11,0:4] #taking the first 10 data as output data
print ('Original data')
print (outputData)
print()
outputData = outputData.drop('label', axis=1, inplace=False) 
print ('After label deletion')
print (outputData)
print()

# Classification
for i in outputData.index:
    newData = outputData.loc[i,'att1':'att3']
    outputData.loc[i,'label'] = (classify (newData, newDataTrain,a))  
outputData['label'] = outputData['label'].astype(np.int64) 
print ('After Classification')

print (outputData)


z1 = dataTrain.loc[1, 'att1':'att3']
z2 = dataTrain.loc[:, 'att1':'label']

sigmas = {0:1, 1:1, 2:1}
z3 = patternLayer(z1, z2, sigmas)
print(z3)