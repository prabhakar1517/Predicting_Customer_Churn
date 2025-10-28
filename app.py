import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# 1️⃣ Load Model and Encoders
# -------------------------------
@st.cache_resource
def load_model_and_encoders():
    model = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, encoders

model, encoders = load_model_and_encoders()

# -------------------------------
# 2️⃣ Streamlit UI
# -------------------------------
st.title("🔮 Customer Churn Prediction App")
st.write("Predict whether a customer will churn using a trained ML model.")

# Example categorical & numeric inputs
# (Adjust names to match your dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tenure = st.number_input("Tenure (in months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)

# -------------------------------
# 3️⃣ Prepare Input Data
# -------------------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "Contract": [contract],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges]
})

# -------------------------------
# 4️⃣ Safe Encoding Function (No Crash)
# -------------------------------
def safe_transform(column_name, series, encoder):
    known_classes = list(encoder.classes_)
    transformed_values = []
    for val in series:
        if val not in known_classes:
            st.warning(f"⚠️ '{val}' not seen in training for '{column_name}'. Using default '{known_classes[0]}'")
            val = known_classes[0]
        transformed_values.append(val)
    # Ensure it's a numpy array before transforming
    return encoder.transform(np.array(transformed_values))

# Safely encode categorical columns
for col in input_data.columns:
    if col in encoders:
        input_data[col] = safe_transform(col, input_data[col], encoders[col])

# -------------------------------
# 5️⃣ Prediction
# -------------------------------
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("🔍 Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ Customer is likely to churn. (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Customer is not likely to churn. (Probability: {probability:.2f})")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# -------------------------------
# 6️⃣ Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn.")
