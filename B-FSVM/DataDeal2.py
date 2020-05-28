import pandas as pd
import numpy as np
import random
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import Precision
from matplotlib import pyplot as plt
import mca,pandas

def get_data(df, lable,processing='standardization'):
    
    X = df.astype('int64')
    X_continue = X.drop(df.columns[9:], axis=1)
    X_discret = X.drop(df.columns[0:9], axis=1)
    #X = np.array(df)
    lable = lable.values
    if processing == 'scaler':
        X_continue = preprocessing.MinMaxScaler().fit_transform(X_continue)
    elif processing == 'standardization':
        X_continue = preprocessing.StandardScaler().fit_transform(X_continue)
    mca_counts = mca.MCA(X_discret)
    #plt.bar(["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18"],mca_counts.L)
    #plt.show()
    X_discret = mca_counts.fs_r_sup(X_discret, 18)
    data = np.append(np.concatenate((X_continue,X_discret),axis=1),lable[:,None],axis=1)
    return data

if __name__ == '__main__':
    
    data = pd.read_csv("../Database_Encodage.csv")
#    data = pd.read_csv("data/Database_label.csv")
#    data = pd.read_csv("data/Database_onehotencoder.csv")
    X = data.drop(['Loan classification'],axis = 1)
    label = data['Loan classification']
    data = get_data(X,label,'standardization')
#    data = get_data(X,label,'scaler','False')
    Train_data,test = train_test_split(data, test_size=0.2,random_state = 42)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]
    
    clf = svm.SVC(C = 3,class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    Precision.precision(y_pred,y_test)
    
    clf = svm.LinearSVC(C=3,class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    Precision.precision(y_pred,y_test)








