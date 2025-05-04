import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import time  


# Configure the page
st.set_page_config(page_title="💧 Water Quality App", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/nigus/Desktop/AI-3/notebook/water_quality_prediction_processed.csv")

@st.cache_resource
def load_model():
    return joblib.load("C:/Users/nigus/Desktop/AI-3/water_quality_pipeline.pkl")

df = load_data()
model = load_model()

# Title and subtitle
st.markdown("""
    <h1 style='text-align: center; color: #0077b6;'>💧 Water Quality Dashboard Made by Group-3</h1>
    <h4 style='text-align: center; color: #023e8a;'>Analyze, Visualize & Predict Water Potability</h4>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Explore", "🤖 Predict"])

# Tab 1: Overview
with tab1:
    st.header("📋 Dataset Overview")
    st.write(df.head())
    st.subheader("🔎 Missing Values")
    st.write(df.isnull().sum())
    st.subheader("📐 Summary Statistics")
    st.write(df.describe())

# Tab 2: Visualization
with tab2:
    st.header("📊 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Distribution Plot")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    col_choice = st.selectbox("Choose a column", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[col_choice], kde=True, color='skyblue', ax=ax2)
    st.pyplot(fig2)

# Tab 3: Explore
with tab3:
    st.header("🔍 Filter & Explore Data")
    col1, col2 = st.columns(2)
    with col1:
        filter_col = st.selectbox("Select column to filter", df.columns)
    with col2:
        value = st.selectbox("Select value", df[filter_col].dropna().unique())

    filtered = df[df[filter_col] == value]
    st.write(f"Showing data for `{filter_col}` = `{value}`")
    st.dataframe(filtered)

# Tab 4: Prediction
with tab4:
    st.header("🤖 Predict Water Potability")

    st.markdown("Use the sliders below to input water characteristics:")
    st.markdown("<h2 style='text-align: center; color: #38a3a5;'>✨ Real-time Water Potability Predictor ✨</h2>", unsafe_allow_html=True)

    if "Potability" in df.columns:
        feature_cols = df.select_dtypes(include='number').drop(columns=["Potability"]).columns
    else:
        feature_cols = df.select_dtypes(include='number').columns[:-1]

    # Sliders for user input
    input_data = {}
    for col in feature_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.slider(
            f"{col}",
            min_value=round(min_val, 2),
            max_value=round(max_val, 2),
            value=round(mean_val, 2)
        )

    if st.button("🚀 Predict Potability"):
        with st.spinner("🔍 Analyzing water quality..."):
            # Create DataFrame from input data
            input_df = pd.DataFrame([input_data])
            
            # Get the model and scaler from the pipeline dictionary
            scaler = model['scaler']
            clf = model['model']
            
            # Scale the input features
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction = clf.predict(scaled_input)[0]
            probability = clf.predict_proba(scaled_input)[0]
            
            time.sleep(1.5)  # Simulate loading
  
        # Show animated feedback
        if prediction == 1:
            st.success(f"✅ Potable Water! Safe to Drink 💧 (Probability: {probability[1]:.2%})")
            st.balloons()
        else:
            st.error(f"❌ Not Potable! Do Not Drink 🚱 (Probability: {probability[0]:.2%})")
            st.snow()  # Snow effect for alert
