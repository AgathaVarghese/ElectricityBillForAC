import streamlit as st
import numpy as np
import joblib

# Set Page Config
st.set_page_config(page_title="AC Electric Bill Predictor", layout="centered")

# Title and Subtitle
st.title("⚡ AC Electric Bill Predictor")
st.write("Polynomial Regression model to predict electricity bill based on AC consumption.")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# User Input
ac_units = st.number_input(
    "Enter AC Electricity Consumption (Units):",
    min_value=0.0,
    max_value=500.0,
    value=50.0,
    step=1.0
)

# Predict Button
if st.button("Calculate Bill"):
    prediction = model.predict(np.array([[ac_units]]))[0]
    st.success(f"Estimated Electric Bill: ₹{prediction:,.2f}")
