import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("customer_churn_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on their details.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Predict Churn"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "Contract": [contract]
    })

    # Apply encoders (if used)
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'Customer will Churn 😞' if prediction == 1 else 'Customer will Stay 😀'}")
