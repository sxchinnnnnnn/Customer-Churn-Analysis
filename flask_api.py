from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("customer_churn_advanced_model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Churn Prediction API is running",
        "endpoint": "/predict"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'Contract',
        'PaymentMethod',
        'InternetService',
        'OnlineSecurity',
        'TechSupport'
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    avg = data['TotalCharges'] / (data['tenure'] + 1)

    sample = pd.DataFrame([{
        'tenure': data['tenure'],
        'MonthlyCharges': data['MonthlyCharges'],
        'TotalCharges': data['TotalCharges'],
        'Contract': data['Contract'],
        'PaymentMethod': data['PaymentMethod'],
        'InternetService': data['InternetService'],
        'OnlineSecurity': data['OnlineSecurity'],
        'TechSupport': data['TechSupport'],
        'AvgChargesPerMonth': avg
    }])

    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    result = "Churn" if prediction == 1 else "Stay"

    return jsonify({
        "prediction": result,
        "churn_probability": round(probability, 4)
    })


if __name__ == "__main__":
    app.run(debug=True)
