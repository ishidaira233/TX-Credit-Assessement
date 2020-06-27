from imblearn.over_sampling import SMOTENC


def upSampling(x_train, y_train):
    x_train, y_train = SMOTENC(
        categorical_features=[a for a in range(9, len(x_train.columns))]
    ).fit_resample(x_train, y_train)
    return x_train, y_train
