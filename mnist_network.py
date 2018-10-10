# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:39:50 2018

@author: 15327522672
"""

import mnist_data
import numpy as np
import matplotlib.pyplot as plt
import mnist_calculate
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

#sigmoid as activate function
def train_onsigmoid(image_train,image_test,label_test):
    loss = []
    train_acc = []
    test_acc = []
    syn0 = 2*np.random.random((784,100)) - 1
    syn1 = 2*np.random.random((100,50)) - 1
    syn2 = 2*np.random.random((50,25)) - 1
    syn3 = 2*np.random.random((25,10)) - 1
    for i in range(1000):
    
        np.random.shuffle(image_train)
        train_list = image_train[:1000,:-1]
        label_list = image_train[:1000,-1:]
        target = np.zeros((1000,10))
        for k in range(1000):
            target[k,int(label_list[k,0])] = 1
            
        for j in range(100):
            
            l1 = 1/(1+np.exp(-(np.dot(train_list,syn0))))  
            l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))   
            l3 = 1/(1+np.exp(-(np.dot(l2,syn2))))  
            l4 = 1/(1+np.exp(-(np.dot(l3,syn3))))  
            
            l4_delta = (target-l4)*(l4*(1-l4))
            l3_delta = l4_delta.dot(syn3.T) * (l3*(1-l3))
            l2_delta = l3_delta.dot(syn2.T) * (l2*(1-l2))   
            l1_delta = l2_delta.dot(syn1.T) * (l1*(1-l1))   
        
            syn3+=(l3.T.dot(l4_delta))*0.0025    
            syn2+=(l2.T.dot(l3_delta))*0.0025    
            syn1 += l1.T.dot(l2_delta)*0.0025
            syn0 += train_list.T.dot(l1_delta)*0.0025
        
        loss_1, train_acc_1,test_acc_1 = mnist_calculate.calLoss(image_train[:, :-1], image_train[:,-1:],image_test,label_test, syn0, syn1, syn2, syn3, 60000)
        loss.append(loss_1)
        train_acc.append(train_acc_1)
        test_acc.append(test_acc_1)

    return syn0,syn1,syn2,syn3,loss,train_acc,test_acc

def train_ontanh(image_train,image_test,label_test):
    
    syn0 = 2*np.random.random((784,100)) - 1
    syn1 = 2*np.random.random((100,50)) - 1
    syn2 = 2*np.random.random((50,25)) - 1
    syn3 = 2*np.random.random((25,10)) - 1
    loss = []
    train_acc = []
    test_acc = []
    for i in range(1000):
    
        np.random.shuffle(image_train)
        train_list = image_train[:1000,:-1]
        label_list = image_train[:1000,-1:]
        target = np.zeros((1000,10))
        for k in range(1000):
            target[k,int(label_list[k,0])] = 2
        target = target -1  
            
        for j in range(100):
            
            l1 = tanh(np.dot(train_list,syn0)) 
            l2 = tanh(np.dot(l1,syn1))   
            l3 =tanh(np.dot(l2,syn2))  
            l4 = tanh(np.dot(l3,syn3))  #forward
            
            l4_delta = (target-l4)*(1-l4**2)
            l3_delta = l4_delta.dot(syn3.T) * (1-l3**2)
            l2_delta = l3_delta.dot(syn2.T) * (1-l2**2)   
            l1_delta = l2_delta.dot(syn1.T) * (1-l1**2)   #backward
        
            syn3+=(l3.T.dot(l4_delta))*0.000045    
            syn2+=(l2.T.dot(l3_delta))*0.000045   
            syn1 += l1.T.dot(l2_delta)*0.000045
            syn0 += train_list.T.dot(l1_delta)*0.000045   #update
            
        loss_1, train_acc_1,test_acc_1 = mnist_calculate.calLoss(image_train[:, :-1], image_train[:,-1:],image_test,label_test, syn0, syn1, syn2, syn3, 60000)
        loss.append(loss_1)
        train_acc.append(train_acc_1)
        test_acc.append(test_acc_1)
    return syn0,syn1,syn2,syn3,loss,train_acc,test_acc

#read minist data
data_train ={}
data_test={}
data_test = mnist_data.fetch_testingset()
data_train = mnist_data.fetch_traingset()

image_train = np.array(data_train['images'])
label_train = np.array(data_train['labels']).reshape(60000,1)
image_test = data_test['images']
label_test = np.array(data_test['labels']).reshape(10000,1)
loss = []
train_acc = []
image_train = np.append(image_train,label_train,axis = 1)

# start to train 
syn0,syn1,syn2,syn3,loss,train_acc,test_acc = train_onsigmoid(image_train,image_test,label_test)
index = range(1,1000)

plt.plot(index,train_acc,label='train_accuracy')
plt.plot(index,test_acc,label='testing_accuracy')
plt.show()


l1 = tanh(np.dot(image_test,syn0))  
l2 = tanh(np.dot(l1,syn1))   
l3 =tanh(np.dot(l2,syn2))  
l4 = tanh(np.dot(l3,syn3))  

accurate = 0
for j in range(10000):
    if np.argmax(l4[j,:]) == label_test[j]:
        accurate = accurate +1
  
accurancy = 0.0
accurancy=accurate/10000