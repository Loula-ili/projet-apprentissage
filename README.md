# TP Apprentissage — Credit Scoring

Projet de machine learning pour la prédiction du risque de crédit. Il comprend l'exploration des données, l'entraînement de modèles de classification et une API de prédiction.

## Structure

```
tp-apprentissage/
├── tpdata.ipynb          # Notebook principal (exploration, entraînement, évaluation)
├── api.py                # API FastAPI pour la prédiction en temps réel
├── utils.py              # Fonctions utilitaires (évaluation, visualisation)
├── credit_scoring.csv    # Dataset d'entrée
└── .gitignore
```

## Modèles entraînés

- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **MLP (réseau de neurones)**
- **Random Forest**

Les modèles sont évalués sur accuracy, recall, precision et AUC-ROC. Le meilleur pipeline est sauvegardé dans `credit_scoring_pipeline.pkl`.

## Installation

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# ou: venv\Scripts\activate    # Windows

pip install fastapi uvicorn scikit-learn numpy pandas matplotlib seaborn
```

## Lancer l'API

```bash
uvicorn api:app --reload
```

L'API est accessible sur `http://localhost:8000`. Documentation interactive : `http://localhost:8000/docs`.

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{
       "Seniority": 9, "Home": 1, "Time": 60, "Age": 30,
       "Marital": 1, "Records": 0, "Job": 1, "Expenses": 73,
       "Income": 129, "Assets": 0, "Debt": 0,
       "Amount": 800, "Price": 846
     }'
```

## Features du dataset

| Feature    | Description                        |
|------------|------------------------------------|
| Seniority  | Ancienneté professionnelle (années)|
| Home       | Type de logement                   |
| Time       | Durée du crédit (mois)             |
| Age        | Âge du client                      |
| Marital    | Situation matrimoniale              |
| Records    | Antécédents de crédit              |
| Job        | Type d'emploi                      |
| Expenses   | Dépenses mensuelles                |
| Income     | Revenu mensuel                     |
| Assets     | Valeur des actifs                  |
| Debt       | Dettes existantes                  |
| Amount     | Montant du crédit demandé          |
| Price      | Prix du bien financé               |
