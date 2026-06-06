import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# Page setup
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("Credit Card Fraud Detection using Isolation Forest")
st.write(
    "This app uses an unsupervised anomaly detection model to identify suspicious credit card transactions."
)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df


df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())


# Basic dataset info
st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", df.shape[0])

with col2:
    st.metric("Total Features", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())


# Separate features and label
X = df.drop("Class", axis=1)
y = df["Class"]


# Sidebar controls
st.sidebar.header("Model Settings")

contamination = st.sidebar.selectbox(
    "Select contamination value",
    [0.001, 0.002, 0.005],
    index=1
)

run_model = st.sidebar.button("Run Isolation Forest")


if run_model:
    st.subheader("Model Output")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    iso_labels = iso_forest.fit_predict(X_scaled)

    # Convert output:
    # -1 = anomaly, 1 = normal
    # Convert into:
    # 1 = suspicious, 0 = normal
    iso_pred = np.where(iso_labels == -1, 1, 0)

    df_result = df.copy()
    df_result["Prediction"] = iso_pred

    normal_count = (df_result["Prediction"] == 0).sum()
    suspicious_count = (df_result["Prediction"] == 1).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Normal Transactions", normal_count)

    with col2:
        st.metric("Suspicious Transactions", suspicious_count)

    st.write("Prediction meaning:")
    st.write("`0 = Normal Transaction`")
    st.write("`1 = Suspicious Transaction`")

    # Show suspicious transactions
    suspicious_df = df_result[df_result["Prediction"] == 1]

    st.subheader("Suspicious Transactions")
    st.dataframe(suspicious_df.head(20))

    # Optional evaluation using Class label
    st.subheader("Basic Evaluation")

    matched_frauds = suspicious_df[suspicious_df["Class"] == 1].shape[0]

    st.write(f"Actual frauds detected by the model: **{matched_frauds}**")
    st.write(
        "Note: The Class column is used here only for evaluation, not for model training."
    )

st.info(
    "Higher contamination values make the model detect more suspicious transactions, "
    "but they may also increase false positives."
)
