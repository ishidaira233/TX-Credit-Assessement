import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#下采样
def lowSampling(df, percent=3/3):
    data1 = df[df[0] == 1]  # 将多数
    data0 = df[df[0] == 0]  # 将少数类别的样本放在data0

    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0]))

#上采样
def upSampling(train):
    X_train, y_train = SMOTE(kind='svm', ratio=1).fit_sample(train[:, 1:], train[:, 0])
    return X_train, y_train


def drawConfusionM(y_pred, y_test,title):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def loadData(SamplingMethode):
    # 读取数据
    train = pd.read_csv("data.csv", header=0)
    # 将数据都变为int型
    for col in train.columns:
        for i in range(1000):
            train[col][i] = int(train[col][i])

    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)

    # 分割为train和test两个数据集
    train, test = train_test_split(train, test_size=0.2)

    if SamplingMethode == 'upSampling':
        # 这里做上采样
        X_train, y_train = upSampling(train)
        y_train = y_train.reshape(len(y_train), 1)
        train = np.append(y_train, X_train, axis=1)
    elif SamplingMethode == 'lowSamoling':
        train = pd.DataFrame(train)
        train = np.array(lowSampling(train))

    return train[:,1:],train[:,0], test[:,1:],test[:,0]


def LDA(X_train, y_train, X_test):
    # solver：一个字符串，指定了求解最优化问题的算法，可以为如下的值。
    # 'svd'：奇异值分解。对于有大规模特征的数据，推荐用这种算法。
    # 'lsqr'：最小平方差，可以结合skrinkage参数。
    # 'eigen' ：特征分解算法，可以结合shrinkage参数。
    lda = LinearDiscriminantAnalysis(solver='svd', store_covariance = True, tol = 0.1)
    y_pred = lda.fit(X_train, y_train).predict(X_test)
    return y_pred



def QDA(X_train, y_train, X_test):
    lda = QuadraticDiscriminantAnalysis(store_covariance = True)
    y_pred = lda.fit(X_train, y_train).predict(X_test)
    return y_pred


def LG(X_train, y_train, X_test):

    classifier = LogisticRegression(penalty='l1',class_weight='balanced',C=0.8)  # 使用类，参数全是默认的
    classifier.fit(X_train, y_train)  # 训练数据来学习，不需要返回值

    y_pred = classifier.predict(X_test)  # 测试数据，分类返回标记

    return y_pred


def KNN(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return y_pred


def RF(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadData('upSampling')
    ypred_lda = LDA(X_train, y_train, X_test)
    drawConfusionM(ypred_lda, y_test,'LDA')

    ypred_qda = QDA(X_train, y_train, X_test)
    drawConfusionM(ypred_qda, y_test,'QDA')

    ypred_lg = LG(X_train, y_train, X_test)
    drawConfusionM(ypred_lg, y_test,'LG')

    ypred_knn = KNN(X_train, y_train, X_test)
    drawConfusionM(ypred_knn, y_test,'KNN')

    ypred_rf = RF(X_train, y_train, X_test)
    drawConfusionM(ypred_rf, y_test,'RF')

