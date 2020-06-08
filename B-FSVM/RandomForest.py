#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:26:47 2020

@author: ishidaira
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import DataDeal2
from Precision import precision
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
#lowsampling
def lowSampling(X,y,style='prototype'):
    if style=='prototype':
        cc = ClusterCentroids(sampling_strategy=3/7,random_state=42)
        X_resampled, y_resampled = cc.fit_sample(X, y)
    elif style=='random':
        rus = RandomUnderSampler(sampling_strategy=1,random_state=42,replacement=True)
        X_resampled, y_resampled = rus.fit_sample(X, y)
    elif style=='edited':
        enn = EditedNearestNeighbours(random_state=42)
        X_resampled, y_resampled = enn.fit_sample(X, y)
    return X_resampled, y_resampled
    
#upsampling
def upSampling(X_train,y_train):
    X_train, y_train = SMOTE(kind='svm').fit_sample(X_train, y_train)
    return X_train, y_train

data = pd.read_csv("../Database_Encodage.csv")
data_loan = data[data["Loan"]==1]
data_overdraft = data[data["Overdraft"]==1]


###########All##################
X = data.drop(['Loan classification'],axis = 1)
#cols = X.columns
#cols = cols.append(pd.Index(['Loan classification']))
label = data['Loan classification']
X_continue = X.drop(X.columns[9:],axis=1)
X_discret = X.drop(X.columns[0:9],axis=1)
data = DataDeal2.get_data(X_continue,X_discret,label,'standardization')
precisionArray = []
X = data[:,:-1]
y = data[:,-1]
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
sss = KFold(n_splits=5, shuffle=True, random_state=42)
for train, test in sss.split(X, y):
    X_test = X[test]
    y_test = y[test]
    X_train = X[train]
    y_train = y[train]
    X_train,y_train = lowSampling(X_train,y_train,'random')
    #X_train,y_train = upSampling(X_train,y_train)
    clf = RandomForestClassifier(random_state=0,n_jobs=-1,class_weight="balanced")
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    precisionArray.append((precision(y_predict,y_test)))
precisionArray = np.round(np.mean(np.array(precisionArray),axis=0),3)
print('bad precision',precisionArray[0],'good precision',precisionArray[1],'type1', precisionArray[2], 'type2', precisionArray[3], 'total accuracy', precisionArray[4],'AUC',precisionArray[5])
#############Loan#############
#data_loan = data_loan.drop(['Duration (in months)','Overdraft'],axis = 1)
#X = data_loan.drop(['Loan classification'],axis = 1)
#label = data_loan['Loan classification']
#X_continue = X.drop(X.columns[8:],axis=1)
#X_discret = X.drop(X.columns[0:8],axis=1)
#data_loan = DataDeal2.get_data(X_continue,X_discret,label,'standardization')
#precisionArray = []
#X = data_loan[:,:-1]
#y = data_loan[:,-1]
#for i in range(len(y)):
#    if y[i] == 0:
#        y[i] = -1
#sss = KFold(n_splits=5, shuffle=True, random_state=42)
#for train, test in sss.split(X, y):
#    X_test = X[test]
#    y_test = y[test]
#    X_train = X[train]
#    y_train = y[train]
#    X_train,y_train = lowSampling(X_train,y_train,'random')
#    #X_train,y_train = upSampling(X_train,y_train)
#    clf = RandomForestClassifier(random_state=0,n_jobs=-1,class_weight="balanced")
#    clf.fit(X_train, y_train)
#    y_predict = clf.predict(X_test)
#    precisionArray.append((precision(y_predict,y_test)))
#precisionArray = np.round(np.mean(np.array(precisionArray),axis=0),3)
#print('bad precision',precisionArray[0],'good precision',precisionArray[1],'type1', precisionArray[2], 'type2', precisionArray[3], 'total accuracy', precisionArray[4],'AUC',precisionArray[5])
#
#############Overdraft#############
#data_overdraft = data_overdraft.drop(['Loan'],axis = 1)
#X = data_overdraft.drop(['Loan classification'],axis = 1)
##cols = X.columns
##cols = cols.append(pd.Index(['Loan classification']))
#label = data_overdraft['Loan classification']
#X_continue = X.drop(X.columns[9:],axis=1)
#X_discret = X.drop(X.columns[0:9],axis=1)
#data_overdraft = DataDeal2.get_data(X_continue,X_discret,label,'standardization')
#precisionArray = []
#X = data_overdraft[:,:-1]
#y = data_overdraft[:,-1]
#for i in range(len(y)):
#    if y[i] == 0:
#        y[i] = -1
#sss = KFold(n_splits=5, shuffle=True, random_state=42)
#for train, test in sss.split(X, y):
#    X_test = X[test]
#    y_test = y[test]
#    X_train = X[train]
#    y_train = y[train]
#    #X_train,y_train = lowSampling(X_train,y_train,'random')
#    X_train,y_train = upSampling(X_train,y_train)
#    clf = RandomForestClassifier(random_state=0,n_jobs=-1,class_weight="balanced")
#    clf.fit(X_train, y_train)
#    y_predict = clf.predict(X_test)
#    precisionArray.append((precision(y_predict,y_test)))
#precisionArray = np.round(np.mean(np.array(precisionArray),axis=0),3)
#print('bad precision',precisionArray[0],'good precision',precisionArray[1],'type1', precisionArray[2], 'type2', precisionArray[3], 'total accuracy', precisionArray[4],'AUC',precisionArray[5])