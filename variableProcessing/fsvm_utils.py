
import numpy as np


def calcul_one_knn_distance(X, X_train, index, K_num):

    distance = []
    for i in range(len(X_train)):
        if i == index:
            continue
        dist = 0
        for (x, y) in zip(X, X_train[i]):
            dist += (x-y)**2

        distance.append(dist**0.5)
    sortedDistance = sorted(distance)[0:K_num]

    distance_avg = np.mean(sortedDistance)
    return distance_avg


def calcul_knn_distance(X_train, K_num):

    X_train_list = X_train.tolist()
    X_distance = []
    for i in range(len(X_train_list)):
        X_distance.append(calcul_one_knn_distance(
            X_train[i], X_train_list, i, K_num))
    X_max = max(X_distance)
    X_min = min(X_distance)
    return X_distance, X_max, X_min


def sort_good_bad(X_train, y_train):
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_list_bad = []
    X_list_good = []
    y_list_bad = []
    y_list_good = []
    count = 0
    for i in range(len(X_train_list)):
        if y_train_list[i] == -1:
            X_list_bad.append(X_train_list[i])
            y_list_bad.append(y_train_list[i])
            count += 1
        else:
            X_list_good.append(X_train_list[i])
            y_list_good.append(y_train_list[i])
    X_list_bad.extend(X_list_good)
    y_list_bad.extend(y_list_good)

    X_train = np.array(X_list_bad)
    y_train = np.array(y_list_bad)
    return X_train, y_train, count


def calcul_avg_distance(X_train):

    X_sum = np.zeros(X_train.shape[1])

    for i in range(0, len(X_train)):
        X_sum += X_train[i]

    X_mean = X_sum / len(X_train)
    distance = []
    for i in range(0, len(X_train)):
        dist = 0
        for (x, y) in zip(X_mean, X_train[i]):
            dist += (x - y) ** 2
        distance.append(dist ** 0.5)
    return distance
