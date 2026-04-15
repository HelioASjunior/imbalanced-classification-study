#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

import pandas as pd
from imblearn.over_sampling import SMOTE


def create_undersampled_dataframe(df, target_col: str = "Class", random_state: int = 42) -> pd.DataFrame:
    fraudes = df[df[target_col] == 1]
    normais = df[df[target_col] == 0].sample(n=len(fraudes), random_state=random_state)
    return pd.concat([fraudes, normais])


def apply_smote(X, y):
    smote = SMOTE()
    return smote.fit_resample(X, y)
