import pandas as pd
import numpy as np

file = "./dataset/data.csv"

# Create a csv with dummies variables


def transformDataBinary(filename):
    df = pd.read_csv(filename, header=0, sep=",",
                     index_col=False,  error_bad_lines=False)

    for col in df:
        if df[col].dtypes == "object":
            dummies = pd.get_dummies(df[col])
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
    print(df)

    df.to_csv(f'{filename[0:-4]}withBinaryData.csv', index=False)


def adaptValue1(n):
    switcher = {
        1: -500,
        2: 100,
        3: 500,
        4: 0
    }
    return switcher.get(n, n)


def adaptValue2(n):
    switcher = {
        1: 50,
        2: 300,
        3: 200,
        4: 750,
        5: -1
    }
    return switcher.get(n, n)


def adaptValue3(n):
    switcher = {
        1: -1,
        2: 0.5,
        3: 2.5,
        4: 5.5,
        5: -8.5
    }
    return switcher.get(n, n)


def transformDataMiddleClass(filename):
    df = pd.read_csv(filename, header=0, sep=",",
                     index_col=False,  error_bad_lines=False)

    print(df.columns)

    df["Account Balance"] = df["Account Balance"].apply(adaptValue1)
    df["Value Savings/Stocks"] = df["Value Savings/Stocks"].apply(adaptValue2)
    df["Length of current employment"] = df["Length of current employment"].apply(
        adaptValue3)
    df.to_csv(f'{filename[0:-4]}withMidClassEncoding2.csv', index=False)


if __name__ == '__main__':
    # transformDataBinary(file)
    transformDataMiddleClass(file)
