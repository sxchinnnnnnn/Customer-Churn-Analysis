import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("customer_churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn risk")

# ================= INPUTS =================
tenure = st.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
)

# Manual encoding based on LabelEncoder alphabetical order
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

# Input array
input_data = np.array([[
    tenure,
    monthly_charges,
    total_charges,
    contract_map[contract],
    payment_map[payment]
]])

# Prediction
if st.button("Predict Churn"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer likely to churn ({probability:.2%} risk)")
    else:
        st.success(f"✅ Customer likely to stay ({1 - probability:.2%} confidence)")