#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:20:34 2020

@author: ishidaira
"""

from cvxopt import matrix
import numpy as np
from numpy import linalg
<<<<<<< HEAD

import cvxopt

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import pandas as pd
from scipy.stats import norm

import seaborn as sns

import matplotlib.pyplot as plt
from Precision import precision
from imblearn.over_sampling import SVMSMOTE
from fsvmClass import HYP_SVM
import DataDeal2
from os import mkdir
from LS_FSVM import *
from variableTransformation import *
from variableReduction import applyPcaWithStandardisation
from variableReduction import applyPcaWithNormalisation

from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours

=======
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
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
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
<<<<<<< HEAD
    return np.exp((-linalg.norm(x - y) ** 2) / (sigma ** 2))

#lowsampling
def lowSampling(X,y,style='prototype'):
    if style=='prototype':
        cc = ClusterCentroids(random_state=42)
        X_resampled, y_resampled = cc.fit_sample(X, y)
    elif style=='random':
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_sample(X, y,replacement=True)
    elif style=='edited':
        enn = EditedNearestNeighbours(random_state=42)
        X_resampled, y_resampled = enn.fit_sample(X, y)
    return X_resampled, y_resampled
    

#upsampling
def upSampling(X_train,y_train):
    X_train, y_train = SMOTE(kind='svm').fit_sample(X_train, y_train)
    return X_train, y_train


def grid_search(X,y,kernel='gaussian'):
    rs = StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=0)
    precisionArray = []
    if kernel == 'gaussian':
        index = 0
        for C in [1,10,100,1000]:
            for sigma in [0.6,0.7,0.8]:
                pre = np.zeros(6)
                for test,train in rs.split(X,y):
                    clf = BFSVM(kernel='gaussian',C=C, sigma = sigma)
                    clf.mvalue(X[train], y[train])
                    clf.fit(X[train], y[train])
                    y_predict = clf.predict(X[test])
                    y_test = np.array(y[test])
                    for i in range(len(y_test)):
                        if y_test[i] == 0:
                            y_test[i] = -1
                    pre += precision(y_predict,y_test)
                precisionArray.append((C,sigma,pre/3))
                index += 1
    return precisionArray
                    
        
class BFSVM(object):
    # initial function
    def __init__(self, kernel=None, fuzzyvalue='Logistic',databalance='origine',a = 4, b = 3, C=None, P=None, sigma=None,ratio_class=None):
        """
        init function
        """
=======
    return np.exp((-linalg.norm(x - y) ** 2) / (2 * (sigma ** 2)))


class BFSVM(object):
    # initial function
    def __init__(self, kernel=None, fuzzyvalue='Logistic',databalance='origine',a = 4, b = 3, C=None, P=None, sigma=None):
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
        self.kernel = kernel
        self.C = C
        self.P = P
        self.sigma = sigma
        self.fuzzyvalue = fuzzyvalue
        self.a = a
        self.b = b
        self.databalance = databalance
        if self.C is not None: self.C = float(self.C)
<<<<<<< HEAD
        self.ratio_class = ratio_class
    
    def mvalue(self, X_train,y_train):
        """
        calculate the membership value
        :param X_train: X train sample
        :param y_train: y train sample
        """
#        print('fuzzy value:', self.fuzzyvalue )
        
#   ########     Methode 1 : FSVM ########
#        clf = HYP_SVM(kernel='polynomial', C=1.5, P=1.5)
#        clf.m_func(X_train,y_train)
#        clf.fit(X_train, y_train)
#        score = clf.project(X_train)-clf.b
#   ########      Methode 2:LSFSVM ########
        kernel_dict = {'type': 'RBF','sigma':0.717}
        fuzzyvalue = {'type':'Cen','function':'Lin'}
        clf = LSFSVM(10,kernel_dict, fuzzyvalue,'o',3/4)
        clf._mvalue(X_train, y_train)
        clf.fit(X_train, y_train)
        clf.predict(X_train)
        score = clf.y_predict-clf.b
#   ########       Methode 3:SVM ########
#        clf = SVC(gamma='scale',class_weight={1:1,-1:17})
#        clf.fit(X_train,y_train)
#        score = clf.decision_function(X_train)
        
=======
    
    def mvalue(self, X_train, X_test,y_train):
#        print('fuzzy value:', self.fuzzyvalue )
        clf = HYP_SVM(kernel='polynomial', C=1.5, P=1.5)
        clf.m_func(X_train,X_test,y_train)
        clf.fit(X_train,X_test, y_train)
        score = clf.project(X_train)-clf.b
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
        
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
<<<<<<< HEAD
#            a = self.a
#            b = self.b
            scoreorg = score
            a = 1
            N_plus = len(y_train[y_train==1])
            sorted(score,reverse=True)
            b = np.mean(score[N_plus-1]+score[N_plus])
            m_value = [1/(np.exp(-a*scoreorg[i]-b)+1) for i in range(len(score))]
            self.m_value = np.array(m_value)
            y_str = []
            for i,y in enumerate(y_train):
                if y==1:
                    y_str.append("positive")
                else:
                    y_str.append("negative")
            m_value = pd.DataFrame(dict(membership=self.m_value,y=y_str))
                
        elif self.fuzzyvalue=='Probit':
            mu = self.mu
            sigma = self.sigma
            self.m_value = norm.cdf((score-mu)/sigma)
        return m_value
    def fit(self, X_train, y):
        """
        use the train samples to fit classifier
        :param X_train: X train sample
        :param y: y train sample
        """
        # extract the number of samples and attributes of train and test
        n_samples, n_features = X_train.shape
=======
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

>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
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
<<<<<<< HEAD
=======
        X_test = np.asarray(X_test)
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908

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
<<<<<<< HEAD
            if self.ratio_class==None:
                tmp3 = np.ones(n_samples) * self.C * (1 - self.m_value)
            else:
                tmp3 = np.ones(n_samples) * self.C * (1 - self.m_value) * self.ratio_class
=======
            tmp3 = np.ones(n_samples) * self.C * (1 - self.m_value)
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
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
        
<<<<<<< HEAD
        #alpha<cm so zeta = 0
        sv_alpha = np.array(list(abs(epsilon+beta-self.C*self.m_value) > 1e-8))
        
        #beta<c(1-m) so mu = 0
        sv_beta = np.array(list(abs(beta - self.C*(1-self.m_value)) > 1e-8))
        
        #print(sv_beta)
        ind = np.arange(len(epsilon))[sv]

        ind_alpha = np.arange(len(epsilon))[sv_alpha]
        
        ind_beta = np.arange(len(epsilon))[sv_beta]
        
        self.epsilon_org = epsilon
        self.epsilon = epsilon[sv]
        epsilon_alpha = epsilon[sv_alpha]
        epsilon_beta = epsilon[sv_beta]
=======
        ind = np.arange(len(epsilon))[sv]
        self.epsilon_org = epsilon
        self.epsilon = epsilon[sv]
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
        self.beta_org = beta
        self.beta = beta[sv]
        self.sv = X_train[sv]
        self.sv_y = y[sv]
        self.sv_yorg = y
        X_train = np.asarray(X_train)
        self.K = self.K[:n_samples,:n_samples]
<<<<<<< HEAD

#         Calculate b 
#        ####methode 1######
#        self.b = 0
#        for n in range(len(self.epsilon)):
#            self.b -= np.sum(self.epsilon * self.K[ind[n], sv])
#        self.b /= len(self.epsilon)
        
#        ####methode 2######
#        b= 0
#        if len(epsilon_alpha): 
#            for n in range(len(epsilon_alpha)):
#                b += 1 
#                b -= np.sum(epsilon_alpha * self.K[ind_alpha[n], sv_alpha])
#                
#        if len(epsilon_beta):
#            for n in range(len(epsilon_beta)):
#                b -= 1 
#                b -= np.sum(epsilon_beta * self.K[ind_beta[n], sv_beta])
#
#        self.b = b/(len(epsilon_alpha)+len(epsilon_beta))
#        print('a',self.b)
#        
#        ####methode 3#######
        
        b_alpha,b_beta= 0,0
        if len(epsilon_alpha): 
            for n in range(len(epsilon_alpha)):
                b_alpha = np.max(1-epsilon_alpha * self.K[ind_alpha[n], sv_alpha])
        if len(epsilon_beta):
            for n in range(len(epsilon_beta)):
                b_beta = np.min(-1-epsilon_beta * self.K[ind_beta[n], sv_beta])
        self.b = -(b_alpha+b_beta)/2
#        ####methode 4#######
#        b_alpha = 0
##        print('a',epsilon_alpha)
##        print('b',epsilon_beta)
#        if len(epsilon_alpha): 
#            for n in range(len(epsilon_alpha)):
#                b_alpha += 1 
#                b_alpha -= np.sum(epsilon_alpha * self.K[ind_alpha[n], sv_alpha])
#            b_alpha /= len(epsilon_alpha)
#
#        b_beta = 0
#        if len(epsilon_beta):
#            for n in range(len(epsilon_beta)):
#                b_beta -= 1 
#                b_beta -= np.sum(epsilon_beta * self.K[ind_beta[n], sv_beta])
#            b_beta /= len(epsilon_beta)
#        if b_alpha and b_beta:
#            self.b = (b_alpha+b_beta)/2
#        else:
#            if b_alpha:
#                self.b = b_alpha
#            else:
#                self.b = b_beta
#        print(self.b)
        # Weight vector
        ######methode 5#######
#        self.b = 0
#        for n in range(len(epsilon_alpha)):
#            self.b += y[sv_alpha]
#            self.b -= np.sum(epsilon_alpha * self.K[ind[n], sv_alpha])
#        for n in range(len(epsilon_beta)):
#            self.b += y[sv_beta]
#            self.b -= np.sum(epsilon_beta * self.K[ind[n], sv_beta])
#        self.b /= (len(epsilon_alpha)+len(epsilon_beta))

=======
        
        # Intercept
        self.b = 0
        for n in range(len(self.epsilon)):
            self.b -= np.sum(self.epsilon * self.K[ind[n], sv])
        self.b /= len(self.epsilon)

        # Weight vector
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
        if self.kernel == 'polynomial' or 'gaussian' or 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.epsilon)):
                self.w += self.epsilon[n] * self.sv[n]
        else:
            self.w = None
<<<<<<< HEAD
    def credit_value(self, X):
        """
        get the decision function ad the credit value
        :param X: X samples to get credit value
        :return: credit value
        """
        if self.w is None:
            return np.dot(X, self.w)
=======

    def project(self, X):
        if self.w is None:
            return np.dot(X, self.w) + self.b
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
        else:
            y_predict = np.zeros(len(X))
            X = np.asarray(X)
            for i in range(len(X)):
                s = 0
                for epsilon,sv in zip(self.epsilon,self.sv):
<<<<<<< HEAD
=======

>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
                    if self.kernel == 'polynomial':
                        s +=  epsilon * polynomial_kernel(sv, X[i], self.P)
                    elif self.kernel == 'gaussian':
                        s += epsilon * gaussian_kernel(X[i], sv, self.sigma)
                    else:
                        s += epsilon * linear_kernel(X[i], sv)
<<<<<<< HEAD
                y_predict[i] = s
            #  print(y_predict[i])
            return y_predict
    # predict function 
    def project(self, X):
        """
        get the geometric margin of hyperplane
        :param X: X sampels
        :retrun: geometric margin of X
        """
        return self.credit_value(X)+self.b
    
    def predict(self, X,ratio=None):
        """
        get prediction of X 
        :param X: test samples
        :param ratio: the posive ratio of all samples
        :return : the prediction y
        """
        if ratio==None:
            return np.sign(self.project(X))
        else:
            credit_value = self.credit_value(X)
            cv =sorted(credit_value,reverse=True)
            cutoff = cv[round(len(X)*ratio)]
            return 2*(credit_value>=cutoff)-1
#        
        

## Test Code for data.scv
#def fsvmTrain(SamplingMethode):
#    train = pd.read_csv("../datawithMidClassEncoding2.csv", header=0)
#    for col in train.columns:
#        for i in range(1000):
#            train[col][i] = int(train[col][i])
#    features = train.columns[1:21]
#    X = train[features]
#    y = train['Creditability']
#    min_max_scaler = preprocessing.MinMaxScaler()
#    X = min_max_scaler.fit_transform(X)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
#    if SamplingMethode == 'upSampling':
#        X_train, y_train = upSampling(X_train,y_train)
#    elif SamplingMethode == 'lowSampling':
#        y_train = np.array(y_train)
#        y_train = y_train.reshape(len(y_train), 1)
#        train = np.append(y_train, np.array(X_train), axis=1)
#        train = pd.DataFrame(train)
#        train = np.array(lowSampling(train))
#        X_train = train[:,1:]
#        y_train = train[:,0]
#
#    X_train = np.asarray(X_train)
#    y_train = np.asarray(y_train)
#    for i in range(len(y_train)):
#        if y_train[i] == 0:
#            y_train[i] = -1
#    clf = BFSVM(kernel='gaussian',C=1000, sigma = 0.7)
#    #clf = BFSVM(kernel='polynomial', C=1.5, P=1.5)
#    clf.mvalue(X_train, y_train)
#    clf.fit(X_train, y_train)
#    
#    y_predict = clf.predict(X_test)
#    y_test = np.array(y_test)
#    for i in range(len(y_test)):
#        if y_test[i] == 0:
#            y_test[i] = -1
#    print(np.mean(y_predict!=y_test))
#    precision(y_predict,y_test)
#
#if __name__ == '__main__':
#    fsvmTrain('lowSampling')


#Test Code for _LSSVMtrain

if __name__ == '__main__':
    
#    data = DataDeal.get_data('../german_numerical.csv')
#    data = pd.read_csv("../labelData.csv", sep=",", header=0)
#    
#    X = np.array(data[data.columns[1:]])
#    y = np.array(data["default"].map({0: -1, 1: 1}))
    data = pd.read_csv("../data/Database_Encodage.csv")
    X = data.drop(['Loan classification'],axis = 1)
    label = data['Loan classification']
    data = DataDeal2.get_data(X,label,'standardization')
    precisionArray = []
    X = data[:,:-1]
    y = data[:,-1]
#    data = pd.read_csv("../processedData.csv", sep=",", header=0)
#    # X = applyPcaWithStandardisation(data[data.columns[1:]], 0.9)
#    X = applyPcaWithNormalisation(data[data.columns[1:]], 0.9)
#    # X = np.array(data[data.columns[1:]])
#    y = np.array(data["default"].map({0: -1, 1: 1}))
#    parameter = grid_search(X,y,kernel='gaussian')
#    print(ok)
    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=12)
    #sss = StratifiedKFold(n_splits=10, random_state=12, shuffle=True)
    for train, test in sss.split(X, y):
        X_test = X[test]
        y_test = y[test]
        X_train = X[train]
        y_train = y[train]
        clf = BFSVM(kernel='gaussian',a=8,b=6,C=10, sigma = 0.7,ratio_class=17)
        #for FSVM optimistic parameters are a=4,b=3,C=100,sigma=0.717
        #for BFSVM optimisitc parmeters are a=8,b=6,C=10,sigma=0.6,0.7/0.8
        #X_train,y_train = lowSampling(X_train,y_train,'prototype')
        mvalue = clf.mvalue(X_train, y_train)
        #print(ok)
        clf.fit(X_train, y_train)
        ratio = len(y_train[y_train==1])/len(y_train)
        y_predict = clf.predict(X_test,ratio)
        precisionArray.append((precision(y_predict,y_test)))
    folder_path = "result/"
    #mkdir(folder_path)      
    np.savetxt(folder_path + "predictions.txt", precisionArray)
    precisionArray = np.round(np.mean(np.array(precisionArray),axis=0),3)
    
    print('bad precision',precisionArray[0],'good precision',precisionArray[1],'type1', precisionArray[2], 'type2', precisionArray[3], 'total accuracy', precisionArray[4],'AUC',precisionArray[5])
    
=======


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
>>>>>>> 64f1cc812ac4295f919cea1ee987b9fd0cdea908
