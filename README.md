# 🧠 Customer Churn Prediction Project

## 🎯 Project Goal
The goal of this project is to predict whether a customer will **churn (leave)** or **stay** based on their demographic, account, and service information.  
This helps telecom companies identify at-risk customers and improve retention strategies.

---

## 📊 Dataset Information
**Dataset Name:** WA_Fn-UseC_-Telco-Customer-Churn.csv  
**Source:** IBM Telco Customer Churn dataset  

### Features:
- Customer demographics (Gender, SeniorCitizen, Partner, Dependents)
- Account information (Tenure, Contract type, Payment method)
- Services (Internet, Phone, Streaming)
- Target variable: **Churn** (Yes/No)

---

## ⚙️ Model Summary
The model used for prediction:
- **Algorithm:** Random Forest / XGBoost (choose your actual model)
- **Encoding:** Label Encoding & One-Hot Encoding for categorical features
- **Balancing:** SMOTE used for handling class imbalance
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score

Files:
- `customer_churn_model.pkl` → Trained model  
- `encoders.pkl` → Encoding objects  
- `model.pkl` → Backup model version  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
