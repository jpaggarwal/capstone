'''
Trains an XGBoost (Extreme Gradient Boosting) model using preprocessed engine data.
XGBoost is one of the most powerful algorithms for tabular data and focuses on
correcting errors from previous decision trees.
'''

import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_processed_data(data_dir='../data/processed'):
    """Loads the preprocessed datasets from the specified directory."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    # 1. Load data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Initialize and Train the Model
    # use_label_encoder=False and eval_metric='logloss' are best practices for modern XGBoost
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 3. Make Predictions
    y_pred = model.predict(X_test)
    
    # 4. Evaluate Performance
    print("\n--- XGBoost Evaluation Results ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Save the trained model
    model_path = '../models/xgboost_model.pkl'
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    train_and_evaluate()
