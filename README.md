# Projeto de Detecção de Anomalias em Cartões de Crédito

Projeto desenvolvido por: Hélio Júnior  
Plataforma de Estudos: DIO - Digital Innovation One

Este projeto aplica técnicas de Machine Learning para detecção de fraude em transações de cartão de crédito usando a base pública hospedada em:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## Objetivo

Comparar modelos e abordagens para classificação de transações fraudulentas, incluindo:
- Regressão Logística
- Random Forest
- Pipeline com threshold customizado
- XGBoost
- Busca de hiperparâmetros com GridSearchCV
- Explicabilidade com SHAP (opcional)

## Estrutura do Projeto

- main.py: atalho para iniciar a aplicação
- run.ps1: script de execução com um comando
- src/main.py: orquestração principal do fluxo
- src/data_processing.py: carga e preparo dos dados
- src/logistic_analysis.py: treino, métricas e gráficos da Regressão Logística
- src/balancing.py: undersampling e SMOTE
- src/random_forest_analysis.py: treino e avaliação de Random Forest
- src/pipeline_analysis.py: pipeline de padronização + Regressão Logística e threshold customizado
- src/xgboost_analysis.py: treino, avaliação e importância de features do XGBoost
- src/hyperparameter_tuning.py: ajuste de hiperparâmetros com GridSearchCV
- src/shap_analysis.py: explicabilidade com SHAP
- .gitignore: arquivos e pastas ignorados pelo Git
- requirements.txt: dependências do projeto
- analise.ipynb: notebook para análises adicionais

## Requisitos

- Python 3.10+
- Ambiente virtual recomendado

## Instalação

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

Os testes atuais validam importação e integridade básica dos módulos.

## CI

Foi adicionado um pipeline de CI em .github/workflows/ci.yml para:
- instalar dependências
- executar pytest automaticamente em push e pull request

## Saídas Geradas

Durante a execução, o projeto:
- Exibe relatórios de classificação no terminal para cada modelo
- Exibe AUC e Average Precision para Regressão Logística
- Gera imagens em graficos/: roc_curve.png, precision_recall_curve.png, xgboost_feature_importance.png e shap_bar.png (se SHAP estiver instalado e funcional)

## Observações

- Se o pacote SHAP não estiver instalado, a etapa de explicabilidade é ignorada automaticamente.
- O backend gráfico está configurado para modo não interativo (Agg), ideal para execução em terminal.

## Melhorias Futuras

- Salvar métricas em arquivos CSV/JSON
- Adicionar comparação visual entre modelos
- Inserir validação cruzada para todos os modelos
- Criar testes unitários de comportamento para cada módulo

## Licença

Projeto para fins educacionais.
