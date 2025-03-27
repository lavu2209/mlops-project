import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from train import load_and_preprocess_data

file_path = "C:\Users\PS\Downloads\merged_stock_sentiment_data.csv"

def train_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "random_forest_stock_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model trained and saved!")

def predict_stock():
    model = joblib.load("random_forest_stock_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    _, X_test, y_test, _, _ = load_and_preprocess_data(file_path)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    return round(accuracy, 4), predictions

if __name__ == "__main__":
    train_model()
