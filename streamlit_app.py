import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import requests

# -------------------------------
# Title
# -------------------------------
st.title("🚗 Car Price Prediction App")

st.write("""
### Predict the Selling Price of a Car
Provide car details in the sidebar to estimate its selling price.
""")

# -------------------------------
# Load Model from GitHub
# -------------------------------
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/Sowndharya617/carpriceprediction/main/model.pkl"  # ✅ Update if path differs
    response = requests.get(url)
    with open("model.pkl", "wb") as f:
        f.write(response.content)
    return pk.load(open("model.pkl", "rb"))

model = load_model()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Car Details")

year = st.sidebar.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2018)
present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=200000, value=30000, step=1000)
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# -------------------------------
# Preprocessing Input
# -------------------------------
if fuel_type == "Petrol":
    Fuel_Type_Petrol = 1
    Fuel_Type_Diesel = 0
elif fuel_type == "Diesel":
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 1
else:
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 0

Seller_Type_Individual = 1 if seller_type == "Individual" else 0
Transmission_Manual = 1 if transmission == "Manual" else 0

current_year = 2025
no_years = current_year - year

# Create input dataframe
input_df = pd.DataFrame({
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Owner': [owner],
    'no_years': [no_years],
    'Fuel_Type_Diesel': [Fuel_Type_Diesel],
    'Fuel_Type_Petrol': [Fuel_Type_Petrol],
    'Seller_Type_Individual': [Seller_Type_Individual],
    'Transmission_Manual': [Transmission_Manual]
})

st.subheader("📋 Input Summary")
st.table(input_df.T)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔮 Predict Selling Price"):
    try:
        prediction = model.predict(input_df)
        output = round(prediction[0], 2)
        st.success(f"💰 Estimated Selling Price: ₹ {output} lakhs")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Sowndharya | Powered by Streamlit")
