import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# Load Model and Encoders
# ------------------------------
@st.cache_resource
def load_model_and_encoders():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        encoders = pickle.load(open("encoders.pkl", "rb"))
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None

model, encoders = load_model_and_encoders()

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ------------------------------
# App Title
# ------------------------------
st.title("📊 Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to **churn** or **stay** based on input features.")

# ------------------------------
# User Input Section
# ------------------------------
st.header("🔹 Enter Customer Details")

# Example features (edit to match your dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)

# ------------------------------
# Prepare Input Data
# ------------------------------
input_dict = {
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "TechSupport": [tech_support],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
}

input_data = pd.DataFrame(input_dict)

# ------------------------------
# Encode Categorical Variables
# ------------------------------
if model is not None and encoders is not None:
    try:
        for col in input_data.columns:
            if col in encoders:
                # Handle unseen categories safely
                input_data[col] = input_data[col].map(lambda s: s if s in encoders[col].classes_ else encoders[col].classes_[0])
                input_data[col] = encoders[col].transform(input_data[col])
    except Exception as e:
        st.error(f"Encoding error: {e}")

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔮 Predict Churn"):
    if model is None:
        st.error("Model not loaded properly. Please check your files.")
    else:
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.error(f"⚠️ The customer is likely to **CHURN**. (Probability: {probability:.2f})")
            else:
                st.success(f"✅ The customer is likely to **STAY**. (Probability: {probability:.2f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Developed by Prabhakar Kumar | Churn Prediction Project")
