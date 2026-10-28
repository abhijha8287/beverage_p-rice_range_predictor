# Beverage Price Range Predictor

Predicts the price range a customer is likely to pay for a beverage from survey answers, with a Streamlit app.

## Why this exists
End-to-end machine learning practice: from a raw survey to a tracked, deployed model, following the provided cleaning, feature-engineering and modelling guides.

## What it does
- Cleans the survey data and fixes logical outliers
- Adds features: age group, consumption-frequency and awareness score, zone affluence score, brand-switching indicator
- Trains and compares Gaussian Naive Bayes, logistic regression, SVM, random forest and XGBoost
- Logs every run to MLflow and saves the most accurate model
- `app.py` takes a customer's answers, applies the same features and predicts the price range

## Project structure
```
beverage_p-rice_range_predictor/
├── file.ipynb                             Cleaning, features, training
├── survey_results.csv                     Raw survey data
├── meta_data.txt                          Column descriptions
├── *_instructions.pdf, *_guide.pdf        Task guides
├── mlruns/                                MLflow run logs
├── best_model.pkl                         Selected model
├── model_columns.pkl                      Expected input columns
├── app.py                                 Streamlit prediction app
└── requirements.txt                       Python dependencies
```

## Getting started
Requirements: Python 3.10+.
```bash
git clone https://github.com/abhijha8287/beverage_p-rice_range_predictor.git
cd beverage_p-rice_range_predictor
pip install -r requirements.txt
streamlit run app.py
```
To see the experiments: `mlflow ui`, then open http://127.0.0.1:5000.

## Tech stack
Python, pandas, scikit-learn, XGBoost, MLflow, Streamlit, joblib

## Notes
Live demo: https://beveragepricerangepredictorbyabhishek.streamlit.app
