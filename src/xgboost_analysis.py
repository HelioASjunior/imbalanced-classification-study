#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, scale_pos_weight: int = 10):
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    return xgb


def evaluate_xgboost(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def plot_feature_importance(model, output_path: str = "graficos/xgboost_feature_importance.png"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

    print("Feature Importances:", model.feature_importances_)
