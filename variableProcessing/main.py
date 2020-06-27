from LSFSVM_class.LS_FSVM import LSFSVM
from bfsvmClass import BFSVM
from BFSVM_class.bfsvmClass import BFSVM
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np
from variableReduction import (
    applyPcaWithStandardisation,
    applyPcaWithNormalisation,
    applyKpcaWithStandardisation,
    applyLlleWithStandardisation,
    applyIcaWithStandardisation,
)
from oversampling import upSampling
from sklearn.ensemble import RandomForestClassifier


def showIndicatorTable(accuracy, sp, se, auc):
    table = []
    table.append(["accuracy: ", f"{accuracy:.3f}"])
    table.append(["specifity: ", f"{sp: .3f}"])
    table.append(["Sensitivity: ", f"{se:.3f}"])
    table.append(["AUC: ", f"{auc:.3f}"])

    print(f"accuracy: {accuracy:.3f}")
    print(f"specifity: {sp: .3f}")
    print(f"Sensitivity: {se:.3f}")
    print(f"AUC: {auc:.3f}")

    fig = plt.figure(dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    tableplt = ax.table(cellText=table, loc="center")
    tableplt.set_fontsize(14)
    tableplt.scale(1, 4)
    ax.axis("off")
    plt.show()


def getIndicatorResult(y_test, y_pred):
    correct = np.sum(y_pred == y_test)
    accuracy = correct / len(y_pred)
    cm = confusion_matrix(y_test, y_pred)

    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    auc = roc_auc_score(y_test, y_pred)

    return (accuracy, sp, se, auc)


def applyLSFSVM(data, dimReductionFunction=None, param1=None, param2=None):
    if dimReductionFunction == None:
        dimReductionFunction = applyPcaWithStandardisation
        if param1 == None:
            param1 = 0.999

    if param2 == None:
        X = dimReductionFunction(data[data.columns[1:]], param1)
    else:
        X = dimReductionFunction(data[data.columns[1:]], param1, param2)

    # print("ncomp", len(X[0]))
    Y = np.array(data["default"].map({0: -1, 1: 1, -1: -1}))

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    kernel_dict = {"type": "LINEAR", "sigma": 0.717}
    fuzzyvalue = {"type": "Cen", "function": "Lin"}

    lsfsvm = LSFSVM(10, kernel_dict, fuzzyvalue, "o", 3 / 4)
    fuzzyMember = lsfsvm._mvalue(x_train, y_train)
    lsfsvm.fit(x_train, y_train)

    y_pred = lsfsvm.predict(x_test)
    y_prob = lsfsvm.predict_prob(x_test)
    decision_function = lsfsvm.decision_function(x_test)

    return y_test, y_pred


def applyBFSVM(data, dimReductionFunction=None, param1=None, param2=None):
    if dimReductionFunction == None:
        dimReductionFunction = applyPcaWithStandardisation
        if param1 == None:
            param1 = 0.999

    X = data[data.columns[1:]]
    Y = np.array(data["default"].map({0: -1, 1: 1, -1: -1}))

    # oversampling
    # X, Y = upSampling(X, Y)

    # dim reduction
    if param2 == None:
        X = dimReductionFunction(X, param1)
    else:
        X = dimReductionFunction(X, param1, param2)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    x_train
    kernel_dict = {"type": "LINEAR", "sigma": 0.717}
    fuzzyvalue = {"type": "Cen", "function": "Lin"}

    bfsvm = BFSVM(kernel="gaussian", a=8, b=6, C=10, sigma=0.7)
    fuzzyMember = bfsvm.mvalue(x_train, y_train)
    bfsvm.fit(x_train, y_train)

    y_pred = bfsvm.predict(x_test)
    # y_prob = bfsvm.predict_prob(x_test)
    # decision_function = bfsvm.decision_function(x_test)

    return y_test, y_pred


def applyRF(data, dimReductionFunction=None, param1=None):
    if dimReductionFunction == None:
        dimReductionFunction = applyPcaWithStandardisation
        if param1 == None:
            param1 = 0.999

    X = data[data.columns[1:]]
    Y = np.array(data["default"].map({0: -1, 1: 1, -1: -1}))

    # oversampling
    # X, Y = upSampling(X, Y)

    # dim reduction
    # if dimReductionFunction != None:
    #     X = dimReductionFunction(X, param1)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    x_train, y_train = upSampling(x_train, y_train)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    return y_test, y_pred


if __name__ == "__main__":
    data1 = pd.read_csv("../dataset/processedData.csv", sep=",", header=0)
    data2 = pd.read_csv("../dataset/labelData.csv", sep=",", header=0)
    data3 = pd.read_csv("../dataset/db.csv", sep=",", header=0)
    #  compare mixEncode and labelEncode
    # for i in range(1, 3):
    #     y_test, y_pred = applyLSFSVM(eval(f"data{i}"))
    #     accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    #     # showIndicatorTable(accuracy, sp, se, auc)
    #     print(f"{i}", accuracy, auc)

    #  compare standardisation vs normalisation before pca
    # y_test1, y_pred1 = applyLSFSVM(data1)
    # accuracy, sp, se, auc = getIndicatorResult(y_test1, y_pred1)
    # print("standardisation:", accuracy, auc)

    # y_test2, y_pred2 = applyLSFSVM(data1, applyPcaWithNormalisation)
    # accuracy, sp, se, auc = getIndicatorResult(y_test2, y_pred2)
    # print("normalisation:", accuracy, auc)

    # compare auc and pourcentage of explained variance conserved
    # y1, y2, x = [], [], []
    # for i in range(1, len(data1.columns)):
    #     y_test, y_pred = applyLSFSVM(data1, param1=i)
    #     accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    #     y1.append(auc)
    #     y2.append(accuracy)
    #     x.append(i)
    #     print(f"result for {i}:", accuracy, auc)

    # plt.xlabel("explained variance")
    # plt.ylabel("auc and accuracy")
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.show()

    #  use kpca to reduce dimensions
    # y1, y2, x = [], [], []
    # for i in range(1, len(data1.columns)):
    #     kernel = {"type": "cosine", "gamma": None, "degree": 2}
    #     y_test, y_pred = applyLSFSVM(
    #         data1,
    #         dimReductionFunction=applyKpcaWithStandardisation,
    #         param1=i,
    #         param2=kernel,
    #     )
    #     accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    #     y1.append(auc)
    #     y2.append(accuracy)
    #     x.append(i)
    #     print(f"result for {i}:", accuracy, auc)

    # plt.xlabel("n component keep")
    # plt.ylabel("auc and accuracy")
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.show()

    #  use lle to reduce dimensions
    # y1, y2, x = [], [], []
    # for i in range(1, len(data1.columns)):
    #     y_test, y_pred = applyLSFSVM(
    #         data1, dimReductionFunction=applyLlleWithStandardisation, param1=i
    #     )
    #     accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    #     y1.append(auc)
    #     y2.append(accuracy)
    #     x.append(i)
    #     print(f"result for {i}:", accuracy, auc)

    # y_test, y_pred = applyLSFSVM(
    #     data1, dimReductionFunction=applyLlleWithStandardisation, param1=20
    # )
    # accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    # showIndicatorTable(accuracy, sp, se, auc)
    # plt.xlabel("n component keep")
    # plt.ylabel("auc and accuracy")
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.show()

    # kernel = {"type": "poly", "gamma": None, "degree": 3}
    # y_test, y_pred = applyLSFSVM(
    #     data1,
    #     dimReductionFunction=applyKpcaWithStandardisation,
    #     param1=37,
    #     param2=kernel,
    # )
    # accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    # showIndicatorTable(accuracy, sp, se, auc)

    # Use BFSVM with data
    # y1, y2, x = [], [], []
    # for i in range(30, len(data1.columns)):
    #     y_test, y_pred = applyBFSVM(data1, param1=i)
    #     accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    #     y1.append(auc)
    #     y2.append(accuracy)
    #     x.append(i)
    #     print(f"result for {i}:", accuracy, auc)

    # plt.xlabel("explained variance")
    # plt.ylabel("auc and accuracy")
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.show()

    # # try ica with data
    # y1, y2, x = [], [], []
    # for i in range(30, len(data1.columns)):
    #     y_test, y_pred = applyLSFSVM(
    #         data1, param1=i, dimReductionFunction=applyIcaWithStandardisation,
    #     )
    #     accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    #     y1.append(auc)
    #     y2.append(accuracy)
    #     x.append(i)
    #     print(f"result for {i}:", accuracy, auc)

    # plt.xlabel("explained variance")
    # plt.ylabel("auc and accuracy")
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.show()

    # use bfsvm with other dataset
    # loan = data3[data3["Loan"] == 1].drop(["Loan", "Overdraft"], axis=1)
    overdraft = data3[data3["Overdraft"] == 1].drop(["Loan", "Overdraft"], axis=1)

    # # loan
    # y_test, y_pred = applyBFSVM(
    #     loan, param1=0.90, dimReductionFunction=applyPcaWithNormalisation
    # )
    # accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    # showIndicatorTable(accuracy, sp, se, auc)
    # print(f"result:", accuracy, auc)

    # #  overdraft
    y_test, y_pred = applyBFSVM(overdraft, param1=0.98)
    accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    showIndicatorTable(accuracy, sp, se, auc)
    print(f"result:", accuracy, auc)

    # random forest
    loan = data3[data3["Loan"] == 1].drop(["Loan", "Overdraft"], axis=1)
    overdraft = data3[data3["Overdraft"] == 1].drop(["Loan", "Overdraft"], axis=1)

    y_test, y_pred = applyRF(overdraft)
    accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    showIndicatorTable(accuracy, sp, se, auc)
    print(f"result:", accuracy, auc)

    y_test, y_pred = applyRF(loan)
    accuracy, sp, se, auc = getIndicatorResult(y_test, y_pred)
    showIndicatorTable(accuracy, sp, se, auc)
    print(f"result:", accuracy, auc)
