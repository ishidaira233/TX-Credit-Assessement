from bfsvmClass import BFSVM
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np
from variableReduction import applyPcaWithStandardisation
from variableReduction import applyPcaWithNormalisation


if __name__ == '__main__':
    # data = pd.read_csv("dataset/labelData.csv", sep=",", header=0)
    data = pd.read_csv("dataset/processedData.csv", sep=",", header=0)
    # X = applyPcaWithStandardisation(data[data.columns[1:]], 0.9)
    X = applyPcaWithNormalisation(data[data.columns[1:]], 0.9)
    # X = np.array(data[data.columns[1:]])
    Y = np.array(data["default"].map({0: -1, 1: 1}))

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    kernel_dict = {'type': 'LINEAR', 'sigma': 0.717}
    fuzzyvalue = {'type': 'Cen', 'function': 'Lin'}

    bfsvm = BFSVM(kernel='gaussian',a=8,b=6,C=10, sigma = 0.7)
    m = bfsvm.mvalue(x_train, y_train)
    bfsvm.fit(x_train, y_train)
    y_pred = bfsvm.predict(x_test)
    #y_prob = bfsvm.predict_prob(x_test)
    #decision_function = bfsvm.decision_function(x_test)

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
    tableplt = ax.table(cellText=table, loc='center')
    tableplt.set_fontsize(14)
    tableplt.scale(1, 4)
    ax.axis('off')
    plt.show()