import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

requirements.txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
pip install matplotlib
pip install streamlit pandas numpy seaborn scikit-learn openpyxl


# Page configuration
st.set_page_config(page_title="Customer Personality Clustering", layout="wide")

# App title
st.title("🧑‍🤝‍🧑 Customer Personality Analysis - Clustering App")

# Load clustering model
with open("DBSCAN.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler (2).pkl", "rb") as file:
    scaler = pickle.load(file)

# File uploader
uploaded_file = st.file_uploader('marketing_campaign.xlsx', type=["xlsx"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Read Excel file
    data = pd.read_excel(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(data.head())

    # Keep a copy of original data
    original_data = data.copy()

    # Drop 'Income' if it exists
    data_processed = data.drop(columns=['Income'], errors='ignore')

    # Scale the data
    data_scaled = scaler.transform(data_processed)

    # Predict clusters using DBSCAN (fit_predict, since DBSCAN doesn't have predict)
    clusters = model.fit_predict(data_scaled)

    # Add cluster labels to data
    original_data['Cluster'] = clusters

    st.subheader("🔍 Clustered Customer Data")
    st.dataframe(original_data)

    # Cluster distribution
    st.subheader("📈 Cluster Distribution")
    cluster_counts = original_data['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # Spending pattern by cluster
    st.subheader("💸 Average Spending on Products per Cluster")
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

    if all(col in original_data.columns for col in mnt_cols):
        cluster_means = original_data.groupby('Cluster')[mnt_cols].mean()
        st.dataframe(cluster_means)

        # Visualizing using heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        plt.ylabel("Cluster")
        plt.xlabel("Product Categories")
        st.pyplot(fig)
    else:
        st.error("❌ Required spending columns not found in the uploaded data.")

else:
    st.info("marketing_campaign.xlsx")
