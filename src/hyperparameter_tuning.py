#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def run_xgboost_grid_search(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [50, 100],
    }

    grid = GridSearchCV(
        XGBClassifier(scale_pos_weight=10, eval_metric="logloss"),
        param_grid,
        scoring="recall",
        cv=3,
    )

    grid.fit(X_train, y_train)
    print("Melhor modelo:", grid.best_params_)
    return grid
