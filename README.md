# Projeto de Deteccao de Anomalias em Cartoes de Credito

Projeto desenvolvido por: Hélio Júnior  
Plataforma de Estudos: DIO - Digital Innovation One

Este projeto aplica tecnicas de Machine Learning para deteccao de fraude em transacoes de cartao de credito usando a base publica hospedada em:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## Objetivo

Comparar modelos e abordagens para classificacao de transacoes fraudulentas, incluindo:
- Regressao Logistica
- Random Forest
- Pipeline com threshold customizado
- XGBoost
- Busca de hiperparametros com GridSearchCV
- Explicabilidade com SHAP (opcional)

## Estrutura do Projeto

- main.py: atalho para iniciar a aplicacao
- run.ps1: script de execucao com um comando
- src/main.py: orquestracao principal do fluxo
- src/data_processing.py: carga e preparo dos dados
- src/logistic_analysis.py: treino, metricas e graficos da Regressao Logistica
- src/balancing.py: undersampling e SMOTE
- src/random_forest_analysis.py: treino e avaliacao de Random Forest
- src/pipeline_analysis.py: pipeline de padronizacao + Regressao Logistica e threshold customizado
- src/xgboost_analysis.py: treino, avaliacao e importancia de features do XGBoost
- src/hyperparameter_tuning.py: ajuste de hiperparametros com GridSearchCV
- src/shap_analysis.py: explicabilidade com SHAP
- .gitignore: arquivos e pastas ignorados pelo Git
- requirements.txt: dependencias do projeto
- analise.ipynb: notebook para analises adicionais

## Requisitos

- Python 3.10+
- Ambiente virtual recomendado

## Instalacao

No PowerShell, dentro da pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Como Executar

```powershell
python .\main.py
```

Ou com um comando dedicado:

```powershell
.\run.ps1
```

## Testes

Executar testes locais:

```powershell
pytest -q
```

Os testes atuais validam importacao e integridade basica dos modulos.

## CI

Foi adicionado pipeline de CI em .github/workflows/ci.yml para:
- instalar dependencias
- executar pytest automaticamente em push e pull request

## Saidas Geradas

Durante a execucao, o projeto:
- Exibe relatorios de classificacao no terminal para cada modelo
- Exibe AUC e Average Precision para Regressao Logistica
- Gera imagens em graficos/: roc_curve.png, precision_recall_curve.png, xgboost_feature_importance.png e shap_bar.png (se SHAP estiver instalado e funcional)

## Observacoes

- Se o pacote SHAP nao estiver instalado, a etapa de explicabilidade e ignorada automaticamente.
- O backend grafico esta configurado para modo nao interativo (Agg), ideal para execucao em terminal.

## Melhorias Futuras

- Salvar metricas em arquivos CSV/JSON
- Adicionar comparacao visual entre modelos
- Inserir validacao cruzada para todos os modelos
- Criar testes unitarios de comportamento para cada modulo

## Licenca

Projeto para fins educacionais.
