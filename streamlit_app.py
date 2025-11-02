import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("💡 Customer Churn Prediction — Demo App")

st.write("""
Predict whether a customer is likely to churn based on their profile and service details.
""")

# -------------------------------------------------
# Load Model from GitHub
# -------------------------------------------------
@st.cache_resource
def load_model():
    # ✅ make sure this path matches your GitHub repo location
    url = "https://raw.githubusercontent.com/Sowndharya617/customer-churn/main/models/rf_churn_model.pkl"
    response = requests.get(url)
    with open("rf_churn_model.pkl", "wb") as f:
        f.write(response.content)
    model = joblib.load("rf_churn_model.pkl")
    return model

model = load_model()

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_single(model, input_df):
    """Predict class (0 or 1) and churn probability."""
    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred[0], float(proba[0])

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("Enter Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", value=70.0, step=1.0)
total = st.sidebar.number_input("Total Charges", value=800.0, step=1.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# -------------------------------------------------
# Build Input DataFrame (must match training features)
# -------------------------------------------------
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

st.subheader("📋 Input Summary")
st.table(input_df.T)

# -------------------------------------------------
# Predict Button
# -------------------------------------------------
if st.button("🔮 Predict Churn"):
    try:
        pred, proba = predict_single(model, input_df)
        if pred == 1:
            st.error(f"⚠️ Churn predicted with probability {proba:.2f}. Recommend retention action.")
        else:
            st.success(f"✅ Customer likely to stay (probability {proba:.2f}).")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Developed by Sowndharya | Powered by Streamlit & Scikit-learn")
