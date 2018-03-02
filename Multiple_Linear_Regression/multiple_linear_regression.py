# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:10:11 2018

@author: Rizky Falih
"""

# Multiple Linear Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values # -1 to remove last coloumn
y = dataset.iloc[:, 4].values # get 4 column

# Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state = 0)