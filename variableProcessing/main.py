from LSFSVM_class.LS_FSVM import LSFSVM
from LSFSVM_class import Precision
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np
from variableReduction import applyPcaWithStandardisation
from variableReduction import applyPcaWithNormalisation


if __name__ == "__main__":
    data = pd.read_csv("dataset/processedData.csv", sep=",", header=0)
    # data = pd.read_csv("dataset/labelData.csv", sep=",", header=0)
    # data = pd.read_csv("dataset/data.csv", sep=",", header=0)
    X = applyPcaWithStandardisation(data[data.columns[1:]], 0.98)
    # X = np.array(data[data.columns[1:]])
    Y = np.array(data["default"].map({0: -1, 1: 1}))

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    kernel_dict = {"type": "LINEAR", "sigma": 0.717}
    fuzzyvalue = {"type": "Cen", "function": "Lin"}

    lsfsvm = LSFSVM(10, kernel_dict, fuzzyvalue, "o", 3 / 4)
    m = lsfsvm._mvalue(x_train, y_train)
    lsfsvm.fit(x_train, y_train)
    y_pred = lsfsvm.predict(x_test)
    y_prob = lsfsvm.predict_prob(x_test)
    decision_function = lsfsvm.decision_function(x_test)

    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    auc = roc_auc_score(y_test, y_pred)

    acc_sum = accuracy
    sp_sum = sp
    se_sum = se
    auc_sum = auc

    table = []
    print(f"accuracy: {accuracy:.3f}")
    table.append(["accuracy: ", f"{accuracy:.3f}"])

    print(f"specifity: {sp: .3f}")
    table.append(["specifity: ", f"{sp: .3f}"])

    print(f"Sensitivity: {se:.3f}")
    table.append(["Sensitivity: ", f"{se:.3f}"])

    print(f"AUC: {auc:.3f}")
    table.append(["AUC: ", f"{auc:.3f}"])

    fig = plt.figure(dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    tableplt = ax.table(cellText=table, loc="center")
    tableplt.set_fontsize(14)
    tableplt.scale(1, 4)
    ax.axis("off")
    plt.show()

    # print('y_prob', y_prob)
    # print(y_pred[y_prob < 0.5])
    # print(y_pred[y_prob > 0.5])
    # print(decision_function)

    # Precision.precision(y_pred, y_test)


# def generateTable(filename):
#     acc_sum = 0
#     sp_sum = 0
#     se_sum = 0
#     f1_sum = 0
#     auc_sum = 0
#     random_state = [10]
#     train = None
#     for j in random_state:
#         train = pd.read_csv(filename, header=0)
#         # train = transformDataBinary(filename)
#         # for col in train.columns:
#         #     print("hey")
#         #     print("d", col[i])
#         #     train[col][i] = int(train[col][i])
#         features = train.columns[1:len(train.columns)]
#         X = train[features]
#         y = train[train.columns[0]]
#         min_max_scaler = preprocessing.MinMaxScaler()
#         X = min_max_scaler.fit_transform(X)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=j)

#         X_train = np.asarray(X_train)
#         y_train = np.asarray(y_train)
#         for i in range(len(y_train)):
#             if y_train[i] == 0:
#                 y_train[i] = -1
#         y_test = np.array(y_test)
#         for i in range(len(y_test)):
#             if y_test[i] == 0:
#                 y_test[i] = -1

#         count = 0
#         X_train, y_train, count = utils.sort_good_bad(X_train, y_train)

#         clf = HYP_SVM(kernel='gaussian', C=1, P=2, sigma=0.8)
#         clf.m_func(X_train, X_test, y_train)
#         clf.fit(X_train, X_test, y_train)
#         y_predict = clf.predict(X_test)

#         correct = np.sum(y_predict == y_test)
#         accuracy = correct / len(y_predict)
#         cm = confusion_matrix(y_test, y_predict)
#         sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#         se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
#         auc = roc_auc_score(y_test, y_predict)

#         acc_sum += accuracy
#         sp_sum += sp
#         se_sum += se
#         auc_sum += auc

#     table = []
#     print("accuracy: ", round(acc_sum / len(random_state), 3))
#     table.append(["accuracy: ", round(acc_sum / len(random_state), 3)])

#     print("specifity: ", round(sp_sum / len(random_state), 3))
#     table.append(["specifity: ", round(sp_sum / len(random_state), 3)])

#     print("Sensitivity: ", round(se_sum / len(random_state), 3))
#     table.append(["Sensitivity: ", round(se_sum / len(random_state), 3)])

#     print("AUC: ", round(auc_sum / len(random_state), 3))
#     table.append(["AUC: ", round(auc_sum / len(random_state), 3)])

#     fig = plt.figure(dpi=80)
#     ax = fig.add_subplot(1, 1, 1)
#     tableplt = ax.table(cellText=table, loc='center')
#     tableplt.set_fontsize(14)
#     tableplt.scale(1, 4)
#     ax.axis('off')
#     plt.show()


# # generateTable('./dataset/data.csv')
# # generateTable('./dataset/german_creditwithBinaryData.csv')
# generateTable('./dataset/datawithMidClassEncoding.csv')
