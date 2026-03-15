import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Models & Transformers ---
# (Using st.cache_resource to avoid reloading on every UI interaction)
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model.h5")
    with open("label_encoder_gender1.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)
    with open("onehot_encoder_geo1.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)
    with open("scaler1.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or preprocessors: {e}")
    st.stop()

# --- 3. App Header ---
st.title("📊 Customer Churn Prediction")
st.markdown("""
Welcome to the Churn Prediction Dashboard. Enter the customer's demographic and account details below to evaluate their likelihood of leaving the bank.
""")
st.divider()

# --- 4. User Inputs (Organized Layout) ---
st.subheader("Customer Profile")

# Using columns to group inputs logically
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🌍 Demographics")
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", min_value=18, max_value=92, value=30)

with col2:
    st.markdown("#### 💰 Financials")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=10)
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

with col3:
    st.markdown("#### 🏦 Account History")
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    
    # UX Improvement: Using Yes/No instead of 1/0
    has_cr_card_input = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member_input = st.selectbox("Is Active Member?", ["Yes", "No"])

# Convert UX inputs back to model logic (1 / 0)
has_cr_card = 1 if has_cr_card_input == "Yes" else 0
is_active_member = 1 if is_active_member_input == "Yes" else 0

st.divider()

# --- 5. Prediction Logic ---
# Center the predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Churn Status", use_container_width=True, type="primary")

if predict_button:
    with st.spinner('Analyzing customer data...'):
        # Simulate slight delay for better UX feel
        time.sleep(0.5) 
        
        # Data Preparation
        gender_encoded = label_encoder_gender.transform([gender])[0]
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
        geo_df = pd.DataFrame(geo_encoded, columns=geo_columns)

        # Base Input Data
        input_df = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        # Combine Data & Scale (Crucial: Order must match training data)
        input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)
        input_scaled = scaler.transform(input_df)

        # Make Prediction
        prediction = model.predict(input_scaled)
        churn_probability = prediction[0][0]

        # --- 6. Output UI ---
        st.subheader("Prediction Results")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Churn Probability", value=f"{churn_probability * 100:.1f}%")
            
        with res_col2:
            if churn_probability > 0.5:
                st.error("⚠️ **High Risk:** The customer is likely to churn.")
                st.progress(float(churn_probability), text="Risk Level")
            else:
                st.success("✅ **Low Risk:** The customer is likely to stay.")
                st.progress(float(churn_probability), text="Risk Level")

        # Debug Expander (Hidden by default to keep UI clean)
        with st.expander("🛠️ View Raw Input Data (For Debugging)"):
            st.dataframe(input_df, use_container_width=True)