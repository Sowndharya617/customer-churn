import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests

st.title("💡 Customer Churn Prediction — Demo App")

# -------------------------------
# Load Model (from GitHub)
# -------------------------------
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/Sowndharya617/customer-churn/main/models/rf_churn_model.pkl"
    response = requests.get(url)
    with open("rf_churn_model.pkl", "wb") as f:
        f.write(response.content)
    return joblib.load("rf_churn_model.pkl")

model = load_model()

# -------------------------------
# Prediction Function
# -------------------------------
def predict_single(model, input_df):
    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred[0], float(proba[0])

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter customer information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", value=70.0, step=1.0)
total = st.sidebar.number_input("Total Charges", value=800.0, step=1.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# -------------------------------
# Build Input DataFrame
# -------------------------------
input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly],
    'TotalCharges': [total],
    'gender_flag': [1 if gender == "Female" else 0],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'InternetService_Fiber optic': [1 if internet == "Fiber optic" else 0],
    'InternetService_No': [1 if internet == "No" else 0],
    'PaymentMethod_Mailed check': [1 if payment == "Mailed check" else 0],
    'PaymentMethod_Bank transfer (automatic)': [1 if payment == "Bank transfer (automatic)" else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment == "Credit card (automatic)" else 0],
})

st.subheader("🧾 Input Features")
st.table(input_df.T)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Churn"):
    try:
        pred, proba = predict_single(model, input_df)
        if pred == 1:
            st.error(f"⚠️ Customer likely to **churn** (probability {proba:.2f}). Recommend retention action.")
        else:
            st.success(f"✅ Customer likely to **stay** (probability {proba:.2f}).")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
