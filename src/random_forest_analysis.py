#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_and_evaluate_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators: int = 50,
    max_depth: int = 10,
    random_state: int = 42,
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    print(classification_report(y_test, rf.predict(X_test)))
    return rf
