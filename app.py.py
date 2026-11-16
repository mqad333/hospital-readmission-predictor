import os
import numpy as np
import pandas as pd
import streamlit as st
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
import pickle

st.set_page_config(page_title="Hospital Readmission Risk Predictor", layout="wide")
st.title("Hospital Readmission Risk Predictor")
st.markdown("Automated 30-day hospital readmission risk triage powered by Orange Data Mining.")

@st.cache_resource
def load_model():
    with open("model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

def build_model_input_table(raw_table):
    # Map columns by exact name into model.domain.attributes so no values become NaN
    src_vars = {v.name: v for v in list(raw_table.domain.attributes) + list(raw_table.domain.metas)}
    n_rows = len(raw_table)
    n_attrs = len(model.domain.attributes)
    X = np.full((n_rows, n_attrs), np.nan)

    for j, target_var in enumerate(model.domain.attributes):
        if target_var.name not in src_vars:
            continue
        s_var = src_vars[target_var.name]
        for i in range(n_rows):
            val = raw_table[i, s_var]
            if np.isnan(float(val)) if isinstance(s_var, ContinuousVariable) else str(val) in ("?", "nan", ""):
                continue
            if isinstance(target_var, DiscreteVariable):
                s_str = str(val).strip()
                if s_str in target_var.values:
                    X[i, j] = target_var.values.index(s_str)
                else:
                    # Handle numeric-coded discrete values like "1" vs "1.0"
                    s_clean = s_str[:-2] if s_str.endswith(".0") else s_str
                    if s_clean in target_var.values:
                        X[i, j] = target_var.values.index(s_clean)
            elif isinstance(target_var, ContinuousVariable):
                try:
                    X[i, j] = float(val)
                except ValueError:
                    pass

    input_domain = Domain(model.domain.attributes)
    return Table.from_numpy(input_domain, X)

def score_orange_table(data):
    aligned_data = build_model_input_table(data)
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
