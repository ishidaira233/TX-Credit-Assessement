import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from fsvmClass import HYP_SVM
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from precision import precision
from bfsvmClass import BFSVM
#下采样
def lowSampling(df, percent=3/3):
    data1 = df[df[0] == 1]  # 将多数
    data0 = df[df[0] == 0]  # 将少数类别的样本放在data0

    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0]))


#上采样
def upSampling(X_train,y_train):
    X_train, y_train = SMOTE(kind='svm').fit_sample(X_train, y_train)
    return X_train, y_train



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

    clf = HYP_SVM(kernel='polynomial', C=1.5, P=1.5)
    clf.m_func(X_train,X_test,y_train)
    clf.fit(X_train,X_test, y_train)

    y_predict = clf.predict(X_test)

    y_test = np.array(y_test)
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1
    cm = confusion_matrix(y_test, y_predict)
    sns.heatmap(cm, annot=True, fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print(np.mean(y_predict!=y_test))
    precision(y_predict,y_test)

    clf = BFSVM(kernel='polynomial', C=1.5, P=1.5)
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





