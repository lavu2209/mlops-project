import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv("C:\Users\PS\Downloads\merged_stock_sentiment_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    features = ["Open", "High", "Low", "Close", "Volume", "Sentiment_Score", "Score", "Comments"]
    
    encoder = LabelEncoder()
    df["Company"] = encoder.fit_transform(df["Company"])
    features.append("Company")

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
