#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:20:34 2020

@author: ishidaira
"""

from cvxopt import matrix
import numpy as np
from numpy import linalg
import cvxopt
from numpy import linalg as LA
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from precision import precision
from imblearn.over_sampling import SVMSMOTE
from fsvmClass import HYP_SVM
# three kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


# param p
def polynomial_kernel(x, y, p=1.5):
    return (1 + np.dot(x, y)) ** p


# param sigmma
def gaussian_kernel(x, y, sigma=1.0):
    # print(-linalg.norm(x-y)**2)
    x = np.asarray(x)
    y = np.asarray(y)
    return np.exp((-linalg.norm(x - y) ** 2) / (2 * (sigma ** 2)))

#lowsampling
def lowSampling(df, percent=3/3):
    data1 = df[df[0] == 1]  # 将多数
    data0 = df[df[0] == 0]  # 将少数类别的样本放在data0

    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0]))


#upsampling
def upSampling(X_train,y_train):
    X_train, y_train = SMOTE(kind='svm').fit_sample(X_train, y_train)
    return X_train, y_train

class BFSVM(object):
    # initial function
    def __init__(self, kernel=None, fuzzyvalue='Logistic',databalance='origine',a = 4, b = 3, C=None, P=None, sigma=None):
        self.kernel = kernel
        self.C = C
        self.P = P
        self.sigma = sigma
        self.fuzzyvalue = fuzzyvalue
        self.a = a
        self.b = b
        self.databalance = databalance
        if self.C is not None: self.C = float(self.C)
    
    def mvalue(self, X_train, X_test,y_train):
#        print('fuzzy value:', self.fuzzyvalue )
        clf = HYP_SVM(kernel='polynomial', C=1.5, P=1.5)
        clf.m_func(X_train,X_test,y_train)
        clf.fit(X_train,X_test, y_train)
        score = clf.project(X_train)-clf.b
        
        if self.fuzzyvalue=='Lin':
            m_value = (score-max(score))/(max(score)-min(score))
        elif self.fuzzyvalue=='Bridge':
            s_up = np.percentile(score,55)
            s_down = np.percentile(score,45)
            m_value = np.zeros((len(score)))
            for i in range(len(score)):
                if score[i]>s_up:
                    m_value[i] = 1
                elif score[i]<=s_down:
                    m_value[i] = 0
                else:
                    m_value[i] = (score[i]-s_down)/(s_up-s_down)
        elif self.fuzzyvalue=='Logistic':
            a = self.a
            b = self.b
            m_value = np.zeros((len(score)))
            for i in range(len(score)):
                m_value[i] = np.exp(a*score[i]+b)/(np.exp(a*score[i]+b)+1)
        self.m_value = m_value

    def fit(self, X_train, X_test, y):
        # extract the number of samples and attributes of train and test
        n_samples, n_features = X_train.shape
        nt_samples, nt_features = X_test.shape

        # initialize a 2n*2n matrix of Kernel function K(xi,xj)
        self.K = np.zeros((2*n_samples, 2*n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'polynomial':
                    self.K[i, j] = polynomial_kernel(X_train[i], X_train[j],self.P)
                elif self.kernel == 'gaussian':
                    self.K[i, j] = gaussian_kernel(X_train[i], X_train[j], self.sigma)
                else:
                    self.K[i, j] = linear_kernel(X_train[i], X_train[j])

            # print(K[i,j])

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        # P = K(xi,xj)
        P = cvxopt.matrix(self.K)

        # q = [-1,...,-1,-2,...,-2]
        q = np.concatenate((np.ones(n_samples) * -1,np.ones(n_samples) * -2))
        q = cvxopt.matrix(q)
        
        #equality constraints
        # A = [1,...,1,0,...,0]
        A = np.concatenate((np.ones(n_samples) * 1,np.zeros(n_samples)))
        
        A = cvxopt.matrix(A)
        
        A = matrix(A, (1, 2*n_samples), 'd')  # changes done
        # b = [0.0]
        b = cvxopt.matrix(0.0)
        
        #inequality constraints
        
        if self.C is None:
            # tmp1 = -1 as diagonal, n*n
            tmp1 = np.diag(np.ones(n_samples) * -1)
            # tmp1 = 2*tmp1 n*2n
            tmp1 = np.hstack((tmp1,tmp1))
            # tmp2 = n*2n, the second matrix n*n diagonal as -1
            tmp2 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.hstack((np.diag(np.zeros(n_samples)),tmp2))
            
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            G = matrix(G, (6*n_samples,2*n_samples), 'd') 
            # h = [0,0,0,...,0] 2n*1
            h = cvxopt.matrix(np.zeros(2*n_samples))

        else:
            # tmp1 = -1 as diagonal, n*n
            tmp1 = np.diag(np.ones(n_samples) * -1)
            # tmp1 = 2*tmp1 n*2n
            tmp1 = np.hstack((tmp1,tmp1))
            # tmp2 = n*2n, the second matrix n*n diagonal as -1
            tmp2 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.hstack((np.diag(np.zeros(n_samples)),tmp2))
            
            tmp3 = np.identity(n_samples)
            # tmp3 = 2*tmp3 n*2n
            tmp3 = np.hstack((tmp3,tmp3))
            # tmp4 = n*2n, the second matrix n*n diagonal as 1
            tmp4 = np.identity(n_samples)
            tmp4 = np.hstack((np.diag(np.zeros(n_samples)),tmp4))
            # G = tmp1,tmp2,tmp3,tmp4 shape 4n*2n
            G = cvxopt.matrix(np.vstack((tmp1,tmp2,tmp3,tmp4)))
            #G = matrix(G, (6*n_samples,2*n_samples), 'd') 
            # h = 4n*1
            tmp1 = np.zeros(2*n_samples)
            tmp2 = np.ones(n_samples) * self.C * self.m_value
            tmp3 = np.ones(n_samples) * self.C * (1 - self.m_value)
            h = cvxopt.matrix(np.hstack((tmp1,tmp2,tmp3)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # print(solution['status'])
        # Lagrange multipliers
        
        # a = [epsilon1,...,epsilonN,beta1,...,betaN]
        a = np.ravel(solution['x'])
        epsilon = a[:n_samples]
        
        beta = a[n_samples:]
        # Support vectors have non zero lagrange multipliers
        
        sv = np.array(list(epsilon+beta > 1e-5) and list(beta > 1e-5))
        
        ind = np.arange(len(epsilon))[sv]
        self.epsilon_org = epsilon
        self.epsilon = epsilon[sv]
        self.beta_org = beta
        self.beta = beta[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        self.sv_yorg = y
        X_train = np.asarray(X_train)
        self.K = self.K[:n_samples,:n_samples]
        
        # Intercept
        self.b = 0
        for n in range(len(self.epsilon)):
            self.b -= np.sum(self.epsilon * self.K[ind[n], sv])
        self.b /= len(self.epsilon)

        # Weight vector
        if self.kernel == 'polynomial' or 'gaussian' or 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.epsilon)):
                self.w += self.epsilon[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            X = np.asarray(X)
            for i in range(len(X)):
                s = 0
                for epsilon,sv in zip(self.epsilon,self.sv):

                    if self.kernel == 'polynomial':
                        s +=  epsilon * polynomial_kernel(sv, X[i], self.P)
                    elif self.kernel == 'gaussian':
                        s += epsilon * gaussian_kernel(X[i], sv, self.sigma)
                    else:
                        s += epsilon * linear_kernel(X[i], sv)


                y_predict[i] = s
            #  print(y_predict[i])
            return y_predict + self.b

    # predict function 
    def predict(self, X):
        return np.sign(self.project(X))

# Test Code for _LSSVMtrain
def fsvmTrain(SamplingMethode):
    train = pd.read_csv("../data.csv", header=0)
    for col in train.columns:
        for i in range(1000):
            train[col][i] = int(train[col][i])
    features = train.columns[1:21]
    X = train[features]
    y = train['Creditability']
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if SamplingMethode == 'upSampling':
        X_train, y_train = upSampling(X_train,y_train)
    elif SamplingMethode == 'lowSampling':
        y_train = np.array(y_train)
        y_train = y_train.reshape(len(y_train), 1)
        train = np.append(y_train, np.array(X_train), axis=1)
        train = pd.DataFrame(train)
        train = np.array(lowSampling(train))
        X_train = train[:,1:]
        y_train = train[:,0]

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1
    clf = BFSVM(kernel='polynomial',C=1.5, P=1.5)
    clf.mvalue(X_train,X_test, y_train)
    clf.fit(X_train,X_test, y_train)
    
    y_predict = clf.predict(X_test)
    y_test = np.array(y_test)
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1
    print(np.mean(y_predict!=y_test))
    precision(y_predict,y_test)

if __name__ == '__main__':
    fsvmTrain('lowSampling')