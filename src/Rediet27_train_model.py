import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import pickle

def load_and_preprocess_data():
    """Load and preprocess the water quality data"""
    # Load dataset (update path as needed)
    df = pd.read_csv("Data\water_quality_prediction.csv")
    
    # Handle missing values - dropping for simplicity (could use imputation)
    df = df.dropna()
    
    return df

def train_and_save_model():
    """Train model and save pipeline"""
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Separate features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save both the scaler and model
    pipeline = {
        'scaler': scaler,
        'model': model
    }
    
    with open('water_quality_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Model and preprocessing pipeline saved successfully")

if __name__ == '__main__':
    train_and_save_model()