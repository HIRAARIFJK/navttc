
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="centered")
st.title("💰 Bank Marketing Term Deposit Prediction")
st.markdown("Enter customer details to predict if they will subscribe to a term deposit.")

st.sidebar.header("📋 Customer Information")

# Sidebar inputs
age = st.sidebar.slider("Age", 18, 95, 30)
job = st.sidebar.selectbox("Job", list(label_encoders["job"].classes_))
marital = st.sidebar.selectbox("Marital Status", list(label_encoders["marital"].classes_))
education = st.sidebar.selectbox("Education", list(label_encoders["education"].classes_))
default = st.sidebar.selectbox("Has Credit in Default?", list(label_encoders["default"].classes_))
housing = st.sidebar.selectbox("Has Housing Loan?", list(label_encoders["housing"].classes_))
loan = st.sidebar.selectbox("Has Personal Loan?", list(label_encoders["loan"].classes_))
contact = st.sidebar.selectbox("Contact Communication Type", list(label_encoders["contact"].classes_))
month = st.sidebar.selectbox("Last Contact Month", list(label_encoders["month"].classes_))
day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)
duration = st.sidebar.slider("Call Duration (in seconds)", 0, 3000, 300)
campaign = st.sidebar.slider("Number of Contacts During Campaign", 1, 50, 1)
pdays = st.sidebar.slider("Days Since Last Contact", -1, 999, -1)
previous = st.sidebar.slider("Previous Contacts", 0, 10, 0)
poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", list(label_encoders["poutcome"].classes_))
balance = st.sidebar.number_input("Account Balance", value=1000)

# Build input dictionary
input_data = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply label encoders
for col in input_df.select_dtypes(include="object").columns:
    if col in label_encoders:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError as e:
            st.error(f"Encoding error in '{col}': {e}")
            st.stop()
    else:
        st.error(f"Missing encoder for: {col}")
        st.stop()

# Align columns to match training order
try:
    input_df = input_df[model.feature_names_in_]
except Exception as e:
    st.error(f"Column alignment failed: {e}")
    st.stop()

# Prediction
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.success(f"✅ Customer will likely SUBSCRIBE (Confidence: {proba:.2%})")
        else:
            st.warning(f"❌ Customer will likely NOT subscribe (Confidence: {proba:.2%})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Optional: show feature importance
if st.checkbox("Show Feature Importances"):
    importances = model.feature_importances_
    features = model.feature_names_in_
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_title("🔍 Feature Importance")
    st.pyplot(fig)
