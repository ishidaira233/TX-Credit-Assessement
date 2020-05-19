import pandas as pd
from sklearn.preprocessing import LabelEncoder

from labels import *


def transformDataHotEncoding(df, labels=None):
    if labels == None:
        labels = df.columns

    for col in labels:
        if df[col].dtypes == "object":
            if len(df[col].unique()) == 2:
                df[col] = LabelEncoder().fit_transform(df[col])
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)

    return df


def transformDataLabelEncoding(df, labels=None, mode="auto"):
    if labels == None:
        labels = df.columns

    for col in labels:
        if mode == "auto":
            df[col] = LabelEncoder().fit_transform(df[col])
        if mode == "manual":
            df[col] = transformTolabel(df[col], col)

    return df


# Create a csv with transformed variables
if __name__ == "__main__":
    quantitativeLabel = [
        "credit_history",
        "purpose",
        "installment_as_income_perc",
        "personal_status_sex",
        "other_debtors",
        "present_res_since",
        "property",
        "other_installment_plans",
        "housing",
        "credits_this_bank",
        "job",
        "people_under_maintenance",
        "telephone",
        "foreign_worker",
    ]
    quantitativeLabelOrdered = ["account_check_status", "savings", "present_emp_since"]

    df = pd.read_csv("./dataset/raw_german_credit.csv", sep=",", header=0)
    df = transformDataHotEncoding(df, quantitativeLabel)
    df = transformDataLabelEncoding(df, labels=quantitativeLabelOrdered, mode="auto")

    df.to_csv("dataset/processedData.csv", index=False)
