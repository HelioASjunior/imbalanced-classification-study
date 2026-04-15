#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_pipeline(X_train, y_train, max_iter: int = 1000):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=max_iter)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_custom_threshold(y_test, y_probs, threshold: float = 0.3):
    y_pred_custom = (y_probs >= threshold).astype(int)
    print(classification_report(y_test, y_pred_custom))
    return y_pred_custom
