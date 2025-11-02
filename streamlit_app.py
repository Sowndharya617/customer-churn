import streamlit as st
import pandas as pd
import joblib

st.title("💡 Customer Churn Prediction App")

# -------------------------------
# Load Model
# -------------------------------
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

# -------------------------------
# Input Section
# -------------------------------
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=5)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=500.0)

# -------------------------------
# Build raw input DataFrame
# -------------------------------
input_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'InternetService': [internet_service],
    'Contract': [contract],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

st.subheader("🧾 Input Features")
st.table(input_df.T)

# -------------------------------
# One-hot encode & align columns
# -------------------------------
input_encoded = pd.get_dummies(input_df)

# Align columns with model’s training columns
try:
    model_columns = model.feature_names_in_
except AttributeError:
    st.error("⚠️ The model doesn't store feature names. Retrain with sklearn ≥1.0 or provide feature names manually.")
    st.stop()

# Reindex to ensure same structure
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# -------------------------------
# Predict
# -------------------------------
if st.button("🔮 Predict Churn"):
    try:
        pred = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer likely to churn (probability: {proba:.2f})")
        else:
            st.success(f"✅ Customer likely to stay (probability: {proba:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
