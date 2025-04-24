
import streamlit as st
import pandas as pd
import numpy as np
import pickle
# import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("🧑\u200d🤝\u200d🧑 Customer Personality Analysis - Clustering App")

# Load clustering model
with open("hierarchical.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Upload CSV file
data_file = st.file_uploader("Upload Customer Data CSV", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)

    # Feature engineering
    df['Age'] = 2025 - df['Year_Birth']
    df['Loyalty_Months'] = pd.to_datetime("2025-01-01") - pd.to_datetime(df['Dt_Customer'])
    df['Loyalty_Months'] = df['Loyalty_Months'].dt.days // 30
    
    df['Total_Spend'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

    df['Total_Accepted_Campaigns'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                                         'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)

    # Map education to ordinal values
    education_order = {
        'Basic': 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    df['Education'] = df['Education'].map(education_order)

    # Transform Marital_Status
    df['Marital_Status'] = df['Marital_Status'].apply(lambda x: 0 if x in ['Married', 'Together'] else 1)

    # Selected features for clustering
    features = [
        'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth', 'Complain',
        'Z_CostContact', 'Z_Revenue', 'Response',
        'Loyalty_Months', 'Age', 'Total_Spend', 'Total_Accepted_Campaigns'
    ]

    X = df[features]

    # Apply scaling
    X_scaled = scaler.transform(X)

    # Predict clusters
    clusters = model.fit_predict(X_scaled)
    df['Cluster'] = clusters

    # Display data with clusters
    st.subheader("Clustered Data")
    st.write(df[['ID', 'Cluster'] + features])

    # Visualize cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv,
        file_name='clustered_customers.csv',
        mime='text/csv',
    )
