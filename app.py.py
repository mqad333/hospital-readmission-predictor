import os
import pandas as pd
import streamlit as st
from Orange.data import Table, Domain
import pickle

st.set_page_config(page_title="Hospital Readmission Risk Predictor", layout="wide")
st.title("Hospital Readmission Risk Predictor")
st.markdown("Automated 30-day hospital readmission risk triage powered by Orange Data Mining.")

@st.cache_resource
def load_model():
    with open("model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

def score_orange_table(data):
    # Use Orange's native domain conversion (triggers Continuize/Impute/Normalize stored in model.domain)
    pred_domain = Domain(model.domain.attributes, metas=data.domain.metas)
    aligned_data = Table.from_table(pred_domain, data)
    probs = model(aligned_data, model.Probs)

    class_values = [str(v).strip() for v in model.domain.class_var.values]
    pos_idx = len(class_values) - 1
    for candidate in ["<30", "1", "Yes", "yes", "True", "true"]:
        if candidate in class_values:
            pos_idx = class_values.index(candidate)
            break

    risk_scores = probs[:, pos_idx]

    col_names = [var.name for var in data.domain.attributes] + [var.name for var in data.domain.metas]
    rows = []
    for inst in data:
        row = [str(inst[var]) for var in data.domain.attributes] + [str(inst[var]) for var in data.domain.metas]
        rows.append(row)
    feat_df = pd.DataFrame(rows, columns=col_names)

    pred_df = pd.DataFrame({
        "Risk Tier": [
            "High Risk" if p >= 0.50 else ("Moderate Risk" if p >= 0.25 else "Low Risk")
            for p in risk_scores
        ],
        "30-Day Readmission Prob (%)": (risk_scores * 100).round(2),
    })

    return pd.concat([pred_df, feat_df], axis=1)

mode = st.radio(
    "Select Data Input Mode:",
    ["Automated Cohort (test_patients.csv)", "Upload Custom Patient CSV"],
    horizontal=True,
)

orange_table = None
if mode == "Automated Cohort (test_patients.csv)":
    if os.path.exists("test_patients.csv"):
        orange_table = Table.from_file("test_patients.csv")
        st.success(f"Loaded built-in validation cohort: test_patients.csv ({len(orange_table)} patients)")
    else:
        st.error("test_patients.csv not found in repository.")
else:
    uploaded = st.file_uploader("Upload a patient CSV file to check 30-day readmission risk.", type=["csv"])
    if uploaded is not None:
        temp_path = "uploaded_temp.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        orange_table = Table.from_file(temp_path)

if orange_table is not None:
    scored_df = score_orange_table(orange_table)
    high_cnt = int((scored_df["Risk Tier"] == "High Risk").sum())
    mod_cnt = int((scored_df["Risk Tier"] == "Moderate Risk").sum())
    low_cnt = int((scored_df["Risk Tier"] == "Low Risk").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(scored_df))
    c2.metric("High Risk (>=50%)", high_cnt)
    c3.metric("Moderate Risk (25-49%)", mod_cnt)
    c4.metric("Low Risk (<25%)", low_cnt)

    st.dataframe(scored_df, use_container_width=True)

    with st.expander("Model Feature Diagnostics"):
        st.write("Model expected features:", [v.name for v in model.domain.attributes][:10])
        st.write("CSV loaded features:", [v.name for v in orange_table.domain.attributes][:10])
