#Projeto desenvolvido por: [Hélio Júnior]
#Plataforma de Estudos: [DIO - Digital Innovation One]

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None


def run_shap_explainability(model, X_test, sample_size: int = 100, output_path: str = "graficos/shap_bar.png"):
    if shap is None:
        print("SHAP não instalado. Etapa de explicabilidade foi ignorada.")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:sample_size])
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return shap_values
