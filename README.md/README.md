# NEXA-LOANLENS-AI — Fairness-Aware Loan Approval System

A machine learning system that predicts loan approvals, detects gender bias,
mitigates it using Reweighing, and explains decisions with counterfactuals.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run in order
```bash
python src/preprocess.py
python src/model.py
python src/fairness.py
python src/explainer.py
streamlit run app.py
```

## Dataset
German Credit Risk — Kaggle. Place as `data/german_credit_data.csv`.
