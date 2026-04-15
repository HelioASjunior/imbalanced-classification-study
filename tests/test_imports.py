#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

import importlib


def test_import_main_entrypoint():
    module = importlib.import_module("main")
    assert hasattr(module, "main")


def test_import_src_modules():
    modules = [
        "src.main",
        "src.data_processing",
        "src.logistic_analysis",
        "src.balancing",
        "src.random_forest_analysis",
        "src.pipeline_analysis",
        "src.xgboost_analysis",
        "src.hyperparameter_tuning",
        "src.shap_analysis",
    ]

    for module_name in modules:
        imported = importlib.import_module(module_name)
        assert imported is not None
