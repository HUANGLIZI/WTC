import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


def andmal20():
    data = pd.read_csv('/input/android-malware-2020-unb/Dropper.csv', delimiter=',', nrows=1000)
    data.dataframeName = 'Dropper.csv'  # [18554, 9442]
    data.drop(data.columns[0], axis=1, inplace=True)
    data.insert(0, column="class", value=1)
    X = data.iloc[:, 1:].values
    Y = data.iloc[:, [0]].values
    # print(X.shape)
    # print(Y.shape)
    X_train, x_test, y_train, y_test = train_test_split(X, Y.ravel(), test_size=0.2, random_state=10)
    x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=315)
    return x_train, y_train, x_val, y_val, x_test, y_test
