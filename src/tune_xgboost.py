'''
Hyperparameter tuning for XGBoost using RandomizedSearchCV.
This script explores multiple combinations of model settings (learning rate, depth, etc.)
to find the absolute best version of the model for this dataset.
'''

import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_processed_data(data_dir='../data/processed'):
    """Loads the preprocessed datasets from the specified directory."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    return X_train, X_test, y_train, y_test

def tune_and_evaluate():
    # 1. Load data
    print("Loading preprocessed data for tuning...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Define the search space (The "Grid")
    # We define ranges for each setting we want to test.
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # 3. Initialize XGBoost
    xgb = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    
    # 4. Initialize Randomized Search
    # n_iter=20: We will try 20 random combinations.
    # cv=3: 3-fold cross-validation for each combination.
    # scoring='f1_macro': We prioritize the F1-score (balance of precision/recall).
    print("Starting Randomized Search (trying 20 combinations, 3-fold CV)...")
    print("Note: This might take a few minutes depending on your CPU.")
    
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    
    # 5. Run the Search
    random_search.fit(X_train, y_train)
    
    # 6. Show Best Parameters
    print("\n--- Tuning Complete! ---")
    print("Best Parameters Found:")
    print(random_search.best_params_)
    
    # 7. Evaluate the Best Model on Test Set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\nAccuracy Score (Best Model): {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred))
    
    # 8. Save the Tuned Model
    model_path = '../models/tuned_xgboost_model.pkl'
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\nTuned model saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    tune_and_evaluate()
