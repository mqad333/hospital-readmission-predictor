import os
import pandas as pd
import streamlit as st
from Orange.data import Table
import pickle

st.set_page_config(page_title="Hospital Readmission Risk Predictor", layout="wide")
st.title("Hospital Readmission Risk Predictor")
st.markdown("Automated 30-day hospital readmission risk triage powered by Orange Data Mining.")

@st.cache_resource
def load_model():
    with open("model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

def score_patients(df):
    temp_csv = "temp_input.csv"
    df.to_csv(temp_csv, index=False)
    data = Table.from_file(temp_csv)
    preds = model(data)
    probs = model(data, model.Probs)

    results = df.copy()
    class_values = list(model.domain.class_var.values)
    idx_30 = class_values.index("<30") if "<30" in class_values else 0
    risk_scores = probs[:, idx_30]

    results["30-Day Readmission Prob (%)"] = (risk_scores * 100).round(2)
    results["Risk Tier"] = [
        "High Risk" if p >= 0.50 else ("Moderate Risk" if p >= 0.25 else "Low Risk")
        for p in risk_scores
    ]
    return results

mode = st.radio(
    "Select Data Input Mode:",
    ["Automated Cohort (test_patients.csv)", "Upload Custom Patient CSV"],
    horizontal=True,
)

target_df = None
if mode == "Automated Cohort (test_patients.csv)":
    if os.path.exists("test_patients.csv"):
        target_df = pd.read_csv("test_patients.csv")
        st.success(f"Loaded built-in validation cohort: test_patients.csv ({len(target_df)} patients)")
    else:
        st.error("test_patients.csv not found in repository.")
else:
    uploaded = st.file_uploader("Upload a patient CSV file to check 30-day readmission risk.", type=["csv"])
    if uploaded is not None:
        target_df = pd.read_csv(uploaded)

if target_df is not None:
    scored_df = score_patients(target_df)
    high_cnt = int((scored_df["Risk Tier"] == "High Risk").sum())
    mod_cnt = int((scored_df["Risk Tier"] == "Moderate Risk").sum())
    low_cnt = int((scored_df["Risk Tier"] == "Low Risk").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(scored_df))
    c2.metric("High Risk (>=50%)", high_cnt)
    c3.metric("Moderate Risk (25-49%)", mod_cnt)
    c4.metric("Low Risk (<25%)", low_cnt)

    st.dataframe(scored_df, use_container_width=True)
