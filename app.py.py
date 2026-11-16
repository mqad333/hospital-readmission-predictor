import pickle
import Orange
import pandas as pd
import streamlit as st

st.title("Hospital Readmission Risk Predictor")
st.write("Upload a patient CSV file to check 30-day readmission risk.")

with open("model.pkcls", "rb") as f:
    model = pickle.load(f)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=["readmitted"], errors="ignore")
    
    for var in model.domain.attributes:
        if var.is_discrete and var.name in df.columns:
            df[var.name] = df[var.name].astype(str)
            
    df.to_csv("temp.csv", index=False)
    data = Orange.data.Table("temp.csv")
    predictions = model(data)
    probs = model(data, model.Probs)
    
    results = pd.DataFrame({
        "Risk Level": [model.domain.class_var.str_val(p) for p in predictions],
        "High Risk %": (probs[:, 0] * 100).round(1),
        "Moderate Risk %": (probs[:, 1] * 100).round(1),
        "Low Risk %": (probs[:, 2] * 100).round(1)
    })
    
    st.dataframe(results)