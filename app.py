import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Run anomaly detection models
def run_anomaly_detection(df):
    # Preprocess data
    x = df.drop(columns=['Class'])
    y = df['Class']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Anomaly Detection Models
    iso_forest = IsolationForest(contamination=0.001)
    iso_forest.fit(x_scaled)
    y_pred_iso = iso_forest.predict(x_scaled)
    y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
    y_pred_lof = lof.fit_predict(x_scaled)
    y_pred_lof = [1 if x == -1 else 0 for x in y_pred_lof]

    return y_pred_iso, y_pred_lof

# Visualization of results
def plot_results(df, y_pred_iso, y_pred_lof):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=df['Amount'], y=df['Time'], hue=y_pred_iso, palette='coolwarm', ax=ax[0])
    ax[0].set_title('Isolation Forest Anomalies')

    sns.scatterplot(x=df['Amount'], y=df['Time'], hue=y_pred_lof, palette='coolwarm', ax=ax[1])
    ax[1].set_title('Local Outlier Factor Anomalies')

    plt.tight_layout()
    st.pyplot(fig)

# Streamlit App
st.title('Credit Card Fraud Detection App')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    y_pred_iso, y_pred_lof = run_anomaly_detection(df)

    st.write("Isolation Forest Anomaly Detection Results:")
    st.write(pd.Series(y_pred_iso).value_counts())

    st.write("Local Outlier Factor Anomaly Detection Results:")
    st.write(pd.Series(y_pred_lof).value_counts())

    st.write("Visualizations of Detected Anomalies:")
    plot_results(df, y_pred_iso, y_pred_lof)
