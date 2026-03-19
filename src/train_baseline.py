'''
Training a baseline Logistic Regression model using preprocessed engine data.
Evaluates performance using accuracy, precision, recall, F1-score, and confusion matrix.
'''

import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
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
    # We use random_state=42 for reproducibility
    print("Training Logistic Regression baseline model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Make Predictions
    y_pred = model.predict(X_test)
    
    # 4. Evaluate Performance
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Save the trained model
    model_path = '../models/baseline_lr_model.pkl'
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    train_and_evaluate()
