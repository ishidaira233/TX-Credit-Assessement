import pandas as pd
import numpy as np

file = "./dataset/german_credit.csv"


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


# def transformDataOther(filename):
#     df = pd.read_csv(filename, header=0, sep=",", index_col=0)

#     print(df)
#     df.to_csv('germanHotEncoding.csv')


if __name__ == '__main__':
    transformDataBinary(file)
