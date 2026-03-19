'''Preprocessing raw engine data: performing stratified train-test split, feature engineering, IQR-based outlier capping, and feature standardization.'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath):
    """Loads the raw dataset."""
    return pd.read_csv(filepath)

def feature_engineering(X):
    """
    Creates derived features to capture physical interactions between sensors.
    These 'physics-based' clues help the model distinguish between healthy/failing states.
    """
    X_engineered = X.copy()
    
    # 1. Thermal Imbalance: Difference between oil and coolant temperatures.
    # In healthy engines, these move together. A large gap is a red flag.
    X_engineered['temp_diff'] = X['lub oil temp'] - X['Coolant temp']
    
    # 2. Engine Stress Index: RPM * Fuel Pressure.
    # High RPM combined with high fuel input indicates high mechanical stress.
    X_engineered['stress_index'] = X['Engine rpm'] * X['Fuel pressure']
    
    # 3. Pressure Ratio: Lub oil pressure vs Coolant pressure.
    # A drop in oil pressure relative to coolant pressure indicates specific pump/leak failures.
    # We add a small epsilon (1e-6) to prevent division by zero.
    X_engineered['pressure_ratio'] = X['Lub oil pressure'] / (X['Coolant pressure'] + 1e-6)
    
    return X_engineered

def handle_outliers(X_train, X_test, columns):
    """
    Caps outliers using the IQR (Interquartile Range) method.
    Limits are calculated on Training data only to prevent leakage.
    """
    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()
    
    for col in columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train_capped[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test_capped[col] = np.clip(X_test[col], lower_bound, upper_bound)
        
    return X_train_capped, X_test_capped

def preprocess_data(filepath):
    # 1. Load Data
    df = load_data(filepath)
    X = df.drop('Engine Condition', axis=1)
    y = df['Engine Condition']
    
    # 2. Train-Test Split (Crucial to do this FIRST)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Feature Engineering (NEW STEP)
    # We create the 'physics clues' before capping and scaling.
    print("Performing feature engineering...")
    X_train_eng = feature_engineering(X_train)
    X_test_eng = feature_engineering(X_test)
    
    # 4. Handle Outliers (Including the new engineered columns)
    continuous_features = X_train_eng.columns.tolist()
    X_train_capped, X_test_capped = handle_outliers(X_train_eng, X_test_eng, continuous_features)
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_capped), 
        columns=X_train_capped.columns, 
        index=X_train_capped.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_capped), 
        columns=X_test_capped.columns, 
        index=X_test_capped.index
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir='../data/processed'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    print(f"All preprocessed files and scaler saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    DATA_PATH = '../data/raw/engine_data.csv'
    print("Starting data preprocessing with feature engineering...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(DATA_PATH)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    print("Preprocessing complete!")
    print(f"New feature count: {X_train.shape[1]} (Was 6)")
    print(f"Sample features: {X_train.columns.tolist()}")
