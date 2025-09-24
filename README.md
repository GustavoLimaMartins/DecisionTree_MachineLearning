# Previsão de Custos Médicos com Árvores de Decisão

Este projeto implementa um **algoritmo de Machine Learning baseado em árvore de decisão** para prever o **custo financeiro médico (charges)** de pacientes com base em características pessoais e hábitos de saúde.

## Objetivo

O objetivo do projeto é fornecer uma previsão do valor esperado de gastos médicos de um paciente a partir de suas informações:

- **Idade** (`age`)  
- **Gênero** (`sex`)  
- **Índice de Massa Corporal (IMC)** (`bmi`)  
- **Número de filhos** (`children`)  
- **Fumante ou não** (`smoker`)  
- **Região** (`region`)  

## Dataset

O modelo foi treinado utilizando um conjunto de dados sintéticos ou real de seguros médicos que inclui as variáveis acima e o valor das despesas médicas (`charges`).

## Metodologia

1. **Pré-processamento dos dados:**  
   - Conversão de variáveis categóricas em numéricas (ex.: `sex`, `smoker`, `region`)  
   - Tratamento de valores ausentes  
   - Normalização opcional de features numéricas  

2. **Divisão treino/teste:**  
   - Separação dos dados em conjuntos de treinamento e teste  

3. **Treinamento do modelo:**  
   - Uso de **Decision Tree Regressor** para aprendizado  
   - Ajuste dos hiperparâmetros como `max_depth` para evitar overfitting  

4. **Avaliação do modelo:**  
   - **Acurácia:** 88%  
   - **MAPE (Mean Absolute Percentage Error):** 27%  

## Resultados

O modelo demonstrou boa capacidade de prever os custos médicos com base nos fatores do paciente. Apesar do MAPE de 27%, a acurácia geral foi alta (88%), indicando que a árvore de decisão consegue capturar padrões relevantes nos dados.

## Uso

```python
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Exemplo de dados de entrada
data = pd.DataFrame({
    "age": [29],
    "sex": ["female"],
    "bmi": [27.5],
    "children": [2],
    "smoker": ["no"],
    "region": ["northwest"]
})

# Supondo que o modelo já esteja treinado e salvo
import joblib
model = joblib.load("decision_tree_model.pkl")

predicted_charge = model.predict(data)
print(f"Custo médico previsto: ${predicted_charge[0]:.2f}")
