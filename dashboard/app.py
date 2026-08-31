import streamlit as st
import requests
import os
import pandas as pd

# Configure the page
st.set_page_config(page_title="Zaalima Analytics Dashboard", page_icon="🔮", layout="wide")

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🔮 Customer Churn & LTV Prediction Engine")
st.markdown("Predict if a customer will churn and estimate their Lifetime Value (LTV).")

st.sidebar.header("Customer Details")
st.sidebar.markdown("Enter the customer's data below:")

# Form for user input
with st.sidebar.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    
    submitted = st.form_submit_button("Predict Churn & LTV")

if submitted:
    # Prepare payload
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with st.spinner("Analyzing Customer Data..."):
        try:
            # Predict Churn
            churn_res = requests.post(f"{API_URL}/predict/churn", json=payload)
            churn_data = churn_res.json()
            
            # Predict LTV
            ltv_res = requests.post(f"{API_URL}/predict/ltv", json=payload)
            ltv_data = ltv_res.json()
            
            with col1:
                st.subheader("⚠️ Churn Prediction")
                if churn_data.get("churn_prediction") == 1:
                    st.error("High Risk: Customer is likely to CHURN!")
                else:
                    st.success("Low Risk: Customer is likely to STAY.")
                
                prob = churn_data.get("churn_probability", 0) * 100
                st.metric("Probability of Churn", f"{prob:.1f}%")
                st.info(f"Risk Level: {churn_data.get('risk_level')}")
                
            with col2:
                st.subheader("💰 Lifetime Value (LTV)")
                st.metric("Predicted LTV", f"${ltv_data.get('predicted_ltv', 0):.2f}")
                
                segment = ltv_data.get('ltv_segment', 'Unknown')
                if "High" in segment:
                    st.success(f"Segment: {segment}")
                elif "Medium" in segment:
                    st.warning(f"Segment: {segment}")
                else:
                    st.error(f"Segment: {segment}")
                    
        except requests.exceptions.ConnectionError:
            st.error(f"Failed to connect to API at {API_URL}. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")
