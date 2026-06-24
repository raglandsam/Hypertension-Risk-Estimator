# Hypertension Risk Estimator

A machine learning web app that estimates a person's likelihood of having high blood pressure (hypertension) based on lifestyle, vitals, and general health indicators.

🔗 **Live app:** https://hypertensionapppy-zwlnbe8lmgupq85juiv8xd.streamlit.app/

## Overview

This project went through two iterations. The first version trained on a synthetic dataset (180k+ rows) and used a hand-built weighted scoring formula in the deployed app, never connecting the trained models to live predictions. While debugging the original results, I found that the dataset itself had effectively zero correlation between any feature and the target (AUC ≈ 0.50 across five different model families), meaning the underlying data had no real predictive signal to learn from, regardless of model choice or tuning.

The project was rebuilt from the data layer up using the **BRFSS 2015 Health Indicators dataset** (real US health survey responses, 70,692 records), with the trained model wired directly into the Streamlit app for live inference.

## Problem

Predict `HighBP` (whether a respondent has been told by a health professional they have high blood pressure) from 17 lifestyle, demographic, and health-history features.

## Dataset

- **Source:** BRFSS 2015 Health Indicators (CDC annual health survey, via Kaggle)
- **Size:** 70,692 records, no missing values
- **Class balance:** 56% HighBP / 44% No HighBP
- **Features:** Age (bucketed), Sex, BMI, Smoker, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Stroke, HighChol, CholCheck, Diabetes

## Approach

1. Stratified 80/20 train-test split
2. Standardized numeric features (Age, BMI, GenHlth, MentHlth, PhysHlth)
3. Trained and compared 5 classifiers with class-balanced weighting (`class_weight='balanced'` / `scale_pos_weight`), so models can't win by predicting the majority class:
   - Logistic Regression
   - Random Forest
   - Linear SVC
   - Naive Bayes
   - XGBoost
4. Evaluated all models on **macro F1** and **AUC**, not just accuracy, since accuracy is misleading on imbalanced classes
5. Selected the best model, saved it with `joblib` alongside the fitted scaler and feature metadata
6. Streamlit app loads the saved model directly and runs real inference on user input (no heuristic scoring formula)

## Results

| Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| **XGBoost (best)** | 0.73 | **0.7274** | 0.798 |
| Logistic Regression | 0.73 | 0.7267 | 0.800 |
| Linear SVC | 0.73 | 0.7254 | 0.800 |
| Random Forest | 0.71 | 0.7028 | 0.764 |
| Naive Bayes | 0.70 | 0.6970 | 0.764 |

XGBoost and Logistic Regression performed almost identically; the simpler linear model captured nearly all the learnable signal in this feature set.

**Top predictors (Random Forest feature importance):**
1. BMI (21.5%)
2. Age (16.3%)
3. Diabetes (9.6%)
4. PhysHlth — days of poor physical health (8.9%)
5. GenHlth — self-rated general health (7.5%)

These align with established clinical risk factors for hypertension.

## What I'd improve next

- Hyperparameter tuning (grid/Bayesian search) on XGBoost and Logistic Regression
- SHAP values for per-prediction explainability instead of global feature importance only
- Calibration check on predicted probabilities (reliability curve)

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Streamlit, Joblib

## Running locally

```bash
pip install -r requirements.txt
python train_model.py      # retrains and saves best_model.pkl, scaler.pkl, model_metadata.pkl
streamlit run hypertension_app.py
```
