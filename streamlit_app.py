import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Customer Churn Prediction App 💡")

# Safely load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rf_churn_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Input section
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Convert inputs into a dataframe
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'InternetService': [internet_service],
    'Contract': [contract],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Predict button
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ The customer is likely to **churn**. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ The customer is likely to **stay**. (Probability: {prob:.2f})")
    except KeyError as e:
        st.error(f"Column mismatch: {e}. Make sure model and input features align.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
