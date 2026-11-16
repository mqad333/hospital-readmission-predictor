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

def align_to_domain(raw_table, target_domain):
    src_vars = {v.name: v for v in list(raw_table.domain.attributes) + list(raw_table.domain.metas)}
    n_rows = len(raw_table)
    t_attrs = list(target_domain.attributes)
    X = np.full((n_rows, len(t_attrs)), np.nan)

    raw_strs = {}
    raw_nums = {}
    for name, s_var in src_vars.items():
        s_list, n_list = [], []
        for i in range(n_rows):
            val = raw_table[i, s_var]
            s = str(val).strip()
            if s.endswith(".0"):
                s = s[:-2]
            s_list.append(s)
            try:
                fv = float(val)
                n_list.append(fv)
            except (ValueError, TypeError):
                n_list.append(np.nan)
        raw_strs[name] = s_list
        raw_nums[name] = n_list

    matched_cols = 0
    for j, t_var in enumerate(t_attrs):
        t_name = t_var.name
        if t_name in src_vars:
            matched_cols += 1
            if isinstance(t_var, DiscreteVariable):
                for i in range(n_rows):
                    s = raw_strs[t_name][i]
                    if s in t_var.values:
                        X[i, j] = t_var.values.index(s)
            elif isinstance(t_var, ContinuousVariable):
                X[:, j] = raw_nums[t_name]
        elif "=" in t_name:
            base_col, exp_val = t_name.split("=", 1)
            if base_col in raw_strs:
                matched_cols += 1
                for i in range(n_rows):
                    X[i, j] = 1.0 if raw_strs[base_col][i] == exp_val else 0.0

    clean_attrs = []
    for v in t_attrs:
        if isinstance(v, DiscreteVariable):
            clean_attrs.append(DiscreteVariable(v.name, values=v.values))
        else:
            clean_attrs.append(ContinuousVariable(v.name))

    clean_domain = Domain(clean_attrs)
    return Table.from_numpy(clean_domain, X), matched_cols

def score_orange_table(data):
    orig_dom = getattr(model, "original_domain", None)
    if orig_dom is not None and len(orig_dom.attributes) > 0:
        aligned_data, matched = align_to_domain(data, orig_dom)
        aligned_data = Table.from_numpy(Domain(orig_dom.attributes), aligned_data.X)
    else:
        aligned_data, matched = align_to_domain(data, model.domain)
        aligned_data = Table.from_numpy(Domain(model.domain.attributes), aligned_data.X)

    probs = model(aligned_data, model.Probs)

    # Index 0 corresponds to the positive 30-day readmission class ("<30")
    pos_idx = 0
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

    return pd.concat([pred_df, feat_df], axis=1), matched

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
    scored_df, matched_cnt = score_orange_table(orange_table)
    high_cnt = int((scored_df["Risk Tier"] == "High Risk").sum())
    mod_cnt = int((scored_df["Risk Tier"] == "Moderate Risk").sum())
    low_cnt = int((scored_df["Risk Tier"] == "Low Risk").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(scored_df))
    c2.metric("High Risk (>=50%)", high_cnt)
    c3.metric("Moderate Risk (25-49%)", mod_cnt)
    c4.metric("Low Risk (<25%)", low_cnt)

    st.dataframe(scored_df, use_container_width=True)
    st.caption(f"Model: {type(model).__name__} | Matched features: {matched_cnt}")
