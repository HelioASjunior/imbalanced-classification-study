#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def load_creditcard_data(url: str = DEFAULT_DATA_URL) -> pd.DataFrame:
    return pd.read_csv(url)


def prepare_features(df: pd.DataFrame):
    scaler = StandardScaler()
    prepared = df.copy()
    prepared["Amount_scaled"] = scaler.fit_transform(prepared[["Amount"]])
    prepared = prepared.drop(["Time", "Amount"], axis=1)

    X = prepared.drop("Class", axis=1)
    y = prepared["Class"]
    return prepared, X, y


def split_train_test(X, y, test_size: float = 0.3, random_state: int = 42):
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
