# Titanic Analytics & Predictive Modeling Module (`/analytics`)

This directory contains the end-to-end data science pipeline for the Titanic dataset, structured into exploratory data analysis (`01_eda.py`) and leak-free predictive modeling (`02_modeling.py`).

---

## Directory Structure

```text
analytics/
├── 01_eda.py                     # Part A: Loading, cleaning, univariate/bivariate EDA, sanity check
├── 02_modeling.py                # Part B: Leak-free preprocessing, classifiers, tuning, regression, artifact test
├── titanic.csv                   # Single committed offline fallback dataset
├── titanic_cleaned.csv           # Cleaned dataset output from Part A
├── models/
│   └── titanic_best_pipeline.joblib # Serialized end-to-end production pipeline
├── plots/
│   ├── task3_univariate_age_fare.png
│   ├── task4_correlation_heatmap.png
│   ├── task5_multivariate_story.png
│   ├── task6_standardization_check.png
│   ├── task9_decision_tree.png
│   ├── task10_model_evaluations.png
│   └── task13_regression_residuals.png
└── README.md                     # Documentation, metrics, and deployment recommendation


