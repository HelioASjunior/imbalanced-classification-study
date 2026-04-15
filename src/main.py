#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from .balancing import apply_smote, create_undersampled_dataframe
from .data_processing import load_creditcard_data, prepare_features, split_train_test
from .hyperparameter_tuning import run_xgboost_grid_search
from .logistic_analysis import (
    evaluate_logistic_regression,
    plot_precision_recall_curve,
    plot_roc_curve,
    train_logistic_regression,
)
from .pipeline_analysis import evaluate_custom_threshold, train_logistic_pipeline
from .random_forest_analysis import train_and_evaluate_random_forest
from .shap_analysis import run_shap_explainability
from .xgboost_analysis import evaluate_xgboost, plot_feature_importance, train_xgboost


def main():
    df = load_creditcard_data()
    df_prepared, X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    logistic_model = train_logistic_regression(X_train, y_train)
    y_probs = evaluate_logistic_regression(logistic_model, X_test, y_test)
    plot_roc_curve(y_test, y_probs)
    plot_precision_recall_curve(y_test, y_probs)

    df_under = create_undersampled_dataframe(df_prepared)
    print(f"Dados após undersampling: {len(df_under)} linhas")

    X_res, y_res = apply_smote(X, y)
    print(f"Dados após SMOTE: {len(X_res)} linhas")
    print(f"Classes após SMOTE: {y_res.value_counts().to_dict()}")

    train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

    train_logistic_pipeline(X_train, y_train)
    evaluate_custom_threshold(y_test, y_probs, threshold=0.3)

    xgb_model = train_xgboost(X_train, y_train)
    evaluate_xgboost(xgb_model, X_test, y_test)
    plot_feature_importance(xgb_model)

    run_xgboost_grid_search(X_train, y_train)
    run_shap_explainability(xgb_model, X_test)


if __name__ == "__main__":
    main()