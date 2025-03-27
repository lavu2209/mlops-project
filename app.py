from flask import Flask, render_template, request
import pandas as pd
import joblib
from model import predict_stock

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    accuracy, predictions = predict_stock()

    # Save predictions to CSV for frontend
    df = pd.read_csv("C:\Users\PS\Downloads\merged_stock_sentiment_data.csv")
    df["Predicted Movement"] = predictions
    df.to_csv("static/predictions.csv", index=False)

    return render_template("result.html", accuracy=accuracy, predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
