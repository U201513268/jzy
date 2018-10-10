# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:38:41 2018

@author: 15327522672
"""

import numpy as np
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def calLoss(image_train, labeltru,image_test, label_test,syn0, syn1, syn2, syn3, vector):

    
    labeltruth = np.zeros((60000, 10))
    
    for i in range(60000):
        labeltruth[i,int(labeltru[i,0])] = 2
        labeltruth = labeltruth -1
    l1 = tanh(np.dot(image_train,syn0)) 
    l2 = tanh(np.dot(l1,syn1))  
    l3 =tanh(np.dot(l2,syn2))  
    labelcal = tanh(np.dot(l3,syn3))  
    
    train_accurancy = 0
    test_acc = 0
    for j in range(60000):
        if np.argmax(labelcal[j,:]) == labeltru[j]:
            train_accurancy = train_accurancy +1
    #train set acc   
    train_acc = 0.0
    train_acc=train_accurancy/60000
    
    #test set predict
    
    labeltesttruth = np.zeros((10000, 10))
    
    for i in range(10000):
        labeltesttruth[i,int(label_test[i,0])] = 2
        labeltesttruth = labeltesttruth -1
    l1 = tanh(np.dot(image_test,syn0))
    l2 = tanh(np.dot(l1,syn1))   
    l3 =tanh(np.dot(l2,syn2))  
    labelcal2 = tanh(np.dot(l3,syn3))  
    #test set acc
    test_accurancy = 0
    for j in range(10000):
        if np.argmax(labelcal2[j,:]) == label_test[j]:
            test_accurancy = test_accurancy +1
      
    test_acc = 0.0
    test_acc=test_accurancy/10000
    
    labelcal = labeltruth - labelcal
    labelcal *= labelcal
    loss = 0.0
    loss = labelcal.sum()
    #print("loss:"+str(loss))
    return loss, train_acc,test_acc