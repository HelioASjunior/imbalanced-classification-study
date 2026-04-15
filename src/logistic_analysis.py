#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def train_logistic_regression(X_train, y_train, max_iter: int = 1000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    y_probs = model.predict_proba(X_test)[:, 1]
    print("AUC:", roc_auc_score(y_test, y_probs))
    print("Average Precision:", average_precision_score(y_test, y_probs))
    return y_probs


def plot_roc_curve(y_test, y_probs, output_path: str = "graficos/roc_curve.png"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_precision_recall_curve(y_test, y_probs, output_path: str = "graficos/precision_recall_curve.png"):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Logistic Regression")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
