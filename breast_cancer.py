#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:14:34 2019

@author: javkhlanenkhbold
"""

import pandas as pd
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt 

#%% Testing accuracy of KNNeighbors 

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer["data"], 
                                                    cancer["target"], 
                                                    stratify = cancer.target,
                                                    random_state = 66)

training_accuracy = []
test_accuracy = []

# Testing values between 1 to 10 
neigbors_settings = np.arange(1,11)

for k in neigbors_settings: 
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    #registering the accuracy of training data 
    training_accuracy.append(clf.score(X_train, y_train))
    #registering the accuracy on any value 
    test_accuracy.append(clf.score(X_test, y_test))

    
plt.figure(figsize =(5,5))
plt.plot(neigbors_settings, training_accuracy, label = "Training accuracy")
plt.plot(neigbors_settings, test_accuracy, label = "Testing accuracy")
plt.ylabel("Accuracy")
plt.xlabel("k neighbors")
plt.legend()
plt.show()