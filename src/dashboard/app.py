import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# 1️⃣ Modell abszolút útvonala
# ---------------------------
MODEL_PATH = r"C:\Users\User\CHURN_project\models\churn_model.pkl"  # ✅ abszolút útvonal

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please run train_model.py first!")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.title("Customer Churn Prediction Dashboard")
st.write("Upload a CSV file with customer data to predict churn.")

# ---------------------------
# 3️⃣ CSV feltöltés
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(data.head())

        # ---------------------------
        # 4️⃣ Predikció
        # ---------------------------
        if st.button("Predict Churn"):
            predictions = model.predict(data)
            data['Churn_Prediction'] = predictions
            st.subheader("Predictions")
            st.dataframe(data)
            st.success("✅ Prediction completed!")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------------------
# 5️⃣ Footer
# ---------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn 1.7.2")
