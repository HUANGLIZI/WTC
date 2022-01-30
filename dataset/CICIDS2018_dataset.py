import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def cicids18():
    df = pd.read_csv('../input/ids-intrusion-csv/02-14-2018.csv')
    # df.head()
    # df.info()
    # df["Label"].value_counts()
    df1 = df[df["Label"] == "Benign"]
    df2 = df[df["Label"] == "FTP-BruteForce"]
    df3 = df[df["Label"] == "SSH-Bruteforce"]
    df = pd.concat([df1, df2, df3], axis=0)  # 1048575 rows × 80 columns
    data = df.dropna()
    target = data["Label"]
    data = data.drop(columns="Label", axis=1)
    data = data.loc[:, (df != 0).any(axis=0)]
    target2 = pd.get_dummies(target)
    data = data.drop(columns="Timestamp", axis=1)
    data["Dst Port"] = pd.DataFrame(data["Dst Port"], dtype=np.float)  # convert to float datatype
    cor_B = data.corrwith(other=target2["Benign"], method='pearson')
    cor_F = data.corrwith(other=target2["FTP-BruteForce"], method='pearson')
    cor_S = data.corrwith(other=target2["SSH-Bruteforce"], method='pearson')
    cor_B.sort_values(ascending=False)
    cor_F.sort_values(ascending=False)
    cor_S.sort_values(ascending=False)
    data = data.drop(columns=["Protocol", "PSH Flag Cnt", "Init Fwd Win Byts", "Flow Byts/s", "Flow Pkts/s"], axis=1)

    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(data)
    categorical_columns = categorical_columns_selector(data)
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(target)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=718)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=315)

    return x_train, y_train, x_val, y_val, x_test, y_test
