# MLOps Project
![Python](https://img.shields.io/badge/Python-3.12.6-3776AB?style=for-the-badge&logo=python&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge) ![Task](https://img.shields.io/badge/Task-Binary%20Classification-1f6feb?style=for-the-badge) ![Dataset](https://img.shields.io/badge/Dataset-19%2C535%20rows-6f42c1?style=for-the-badge) ![Features](https://img.shields.io/badge/Features-6%20sensors-8250df?style=for-the-badge) ![Saved Models](https://img.shields.io/badge/Saved%20Models-4-16a34a?style=for-the-badge) ![Best Model](https://img.shields.io/badge/Best%20Model-Tuned%20XGBoost-0ea5e9?style=for-the-badge) ![Accuracy](https://img.shields.io/badge/Accuracy-0.6652-15803d?style=for-the-badge) ![Stage](https://img.shields.io/badge/Stage-Wrapped-f97316?style=for-the-badge)



---

## Project Overview

This project predicts Engine Condition (0/1) from engine sensor readings.  
EDA was done to identify key signals, class overlap, and anomalies, followed by preprocessing (stratified split, feature engineering, outlier capping, scaling).  
We compared Logistic Regression, Random Forest, XGBoost, and tuned XGBoost, with tuned XGBoost performing best.  

---

## Dataset Summary

- File: The model was trained on proprietary engine sensor data.
- Rows: 19,535
- Columns: 7
- Missing values: 0
- Duplicate rows: 0
- Class distribution:
  - Class 1: 12,317 (63.05%)
  - Class 0: 7,218 (36.95%)

---

## Project Structure
```text
mlops-project/
├── data/
│   ├── raw/
│   │   └── engine_data.csv
│   └── processed/                  (generated after preprocessing)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── scaler.pkl
├── models/                         (generated after training/tuning)
│   ├── baseline_lr_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── tuned_xgboost_model.pkl
├── notebooks/
│   └── eda2.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_baseline.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   └── tune_xgboost.py
├── requirements.txt
├── LICENSE
└── README.md
```
---


## EDA Highlights

- Engine rpm is the strongest predictive feature.
- Fuel pressure provides secondary separation.
- Coolant pressure is weak as a standalone predictor.
- Significant class overlap exists in normal operating ranges.
- Rare anomalies exist (high coolant temperature, near-zero fuel pressure), but they are not perfectly deterministic.

---

## Preprocessing Pipeline

- Stratified train-test split
- Feature engineering:
  - temp_diff
  - stress_index
  - pressure_ratio
- IQR-based outlier capping
- Feature standardization

---

## Models Evaluated

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Tuned XGBoost (best)

---

## Results

- Logistic Regression: Accuracy 0.6578, Macro F1 0.58
- Random Forest: Accuracy 0.6537, Macro F1 0.60
- XGBoost: Accuracy 0.6611, Macro F1 0.60
- Tuned XGBoost(Best Performance): Accuracy 0.6652, Macro F1 0.61

Class-wise pattern in best model:
- Better performance on class 1
- Lower recall on class 0

---

## Key Limitation

Performance is capped mainly by class overlap in feature space.
Model improvements are incremental unless better signal/features are added.

---

## Notes

- Processed files and model files are generated automatically after running scripts.

---

## License

This project is licensed under the MIT License.
