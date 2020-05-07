import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def showExplainedVariance(data, threshold=0.9):
    (m, n) = data.shape

    x = []
    y = []
    for i in range(1, n):
        pca = PCA(n_components=i)
        pca.fit_transform(data)
        x.append(i)
        y.append(np.sum(pca.explained_variance_ratio_))

    plt.xlabel('number of component')
    plt.ylabel('explained variance')
    plt.plot(x, y)
    plt.hlines(threshold, xmin=0, xmax=n)

    plt.show()


def applyPcaWithStandardisation(data, threshold=0.9):
    X = preprocessing.scale(data)

    pca = PCA(n_components=threshold)
    return pca.fit_transform(X)


def applyPcaWithNormalisation(data, threshold=0.9):
    X = preprocessing.MinMaxScaler().fit_transform(data)

    pca = PCA(n_components=threshold)
    return pca.fit_transform(X)


if __name__ == '__main__':
    df = pd.read_csv("dataset/processedData.csv", header=0, sep=",")
    df = df.drop("default", axis=1)

    X1 = preprocessing.scale(df)
    showExplainedVariance(X1, 0.8)

    X2 = preprocessing.MinMaxScaler().fit_transform(df)
    showExplainedVariance(X2, 0.8)
