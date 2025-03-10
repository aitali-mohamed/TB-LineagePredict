import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import tempfile
import pysam
import time
import scipy.sparse as sp
from io import BytesIO
import base64
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="TB-LineagePredict", layout="wide")

# ---- CUSTOM CSS FOR STYLING ----
st.markdown("""
    <style>
        .navbar {
            background-color: #2C3E50;
            padding: 15px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            position: fixed;
            z-index: 1000000000;
            left: 0;
            right: 0;
            top: 0;
            height: 70px;
        }
        .center-content {
            max-width: 800px;
            margin: auto;
            text-align: center;
        }
        .flowchart-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }
        .flowchart-step {
            padding: 12px 20px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
            color: white;
            width: 200px;
            transition: background-color 0.5s ease;
        }
        .active {
            background-color: #27ae60; /* Green for active step */
        }
        .inactive {
            background-color: #7f8c8d; /* Grey for inactive steps */
        }
    </style>
""", unsafe_allow_html=True)

def get_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64("assets/LOGO.png")

# ---- NAVIGATION BAR ----
header_bg = "#2C3E50"
accent_color = "#E67E22"
st.markdown(f"""
    <div style="
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background-color: {header_bg}; 
        padding: 15px; 
        border-radius: 5px;
        position: fixed;
        z-index: 1000000000;
        left: 0;
        right: 0;
        top: 0;
        height: 70px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <img src="data:image/png;base64,{logo_base64}" style="height: 60px;" >
            <span style="color: white; font-size: 22px; font-weight: bold;">TB-LineagePredict</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---- CENTERED CONTENT ----
with st.container():
    st.markdown('<style>div.block-container{padding-top: 70px;}</style>', unsafe_allow_html=True)
    st.markdown("## TB-LineagePredict: A Machine Learning Pipeline for MTB Lineage Prediction")
    st.markdown("""
        Welcome to **TB-LineagePredict**, an advanced tool for **Mycobacterium tuberculosis lineage classification**.
        This web-based platform allows users to upload **VCF or TXT files** and run **ML-based predictions**.
    """)

    st.markdown("#### Upload Your Files")
    uploaded_files = st.file_uploader("Upload VCF or TXT files", type=["txt", "vcf", "gz"], accept_multiple_files=True)

# ---- MODEL LOADING ----
@st.cache(allow_output_mutation=True)
def load_models(svd_path, model_path):
    svd = joblib.load(svd_path)
    model = joblib.load(model_path)
    return svd, model

svd_path = "models/truncated_svd.pkl"
model_path = "models/random_forest.pkl"
svd, model = load_models(svd_path, model_path)

# ---- FUNCTION TO DISPLAY & UPDATE FLOWCHART ----
flowchart_placeholder = st.empty()

def update_flowchart(step):
    steps = ["Data Acquisition", "Feature Extraction", "Dimensionality Reduction", "Classification", "Results"]
    step_colors = ["inactive"] * len(steps)
    step_colors[step] = "active"

    flowchart_html = "<div class='flowchart-container'>"
    for i, s in enumerate(steps):
        flowchart_html += f'<div class="flowchart-step {step_colors[i]}">{s}</div>'
    flowchart_html += "</div>"

    flowchart_placeholder.markdown(flowchart_html, unsafe_allow_html=True)

# ---- PROCESS FILES FUNCTION ----
def process_files(uploaded_files):
    processed_data = []
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        if file_ext in [".vcf", ".gz"]:
            mutations = extract_mutations_vcf(uploaded_file)
        elif file_ext == ".txt":
            mutations = extract_mutations_txt(uploaded_file)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            continue
        processed_data.append(mutations)

    df = pd.DataFrame(processed_data).fillna(0)
    return df

# ---- EXTRACT MUTATIONS ----
def extract_mutations_txt(content):
    mutations = {}
    for line in content:
        if line.startswith("CHROM"):
            continue  # Skip header
        parts = line.strip().split("\t")
        if len(parts) > 1:
            pos = int(parts[1])
            mutations[pos] = 1  # Mark SNP presence
    return mutations

def extract_mutations_vcf(uploaded_file):
    mutations = {}
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as temp_vcf:
            temp_vcf.write(uploaded_file.read())
            temp_vcf.flush()
            vcf = pysam.VariantFile(temp_vcf.name)
            for record in vcf:
                mutations[record.pos] = 1
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
    return mutations

# ---- PROCESS & PREDICT ----
if uploaded_files:
    update_flowchart(0)
    time.sleep(1)

    st.write("Extracting features from uploaded files...")
    update_flowchart(1)
    X_new = process_files(uploaded_files)
    X_new_sparse = sp.csr_matrix(X_new.values)
    time.sleep(1)

    st.write("Applying TruncatedSVD for dimensionality reduction...")
    update_flowchart(2)
    X_new_pca = svd.transform(X_new_sparse)
    time.sleep(1)

    st.write("Running classification...")
    update_flowchart(3)
    predictions = model.predict(X_new_pca)
    time.sleep(1)

    # Create the DataFrame for results
    prediction_data = [[uploaded_file.name, pred] for uploaded_file, pred in zip(uploaded_files, predictions)]
    prediction_df = pd.DataFrame(prediction_data, columns=["File Name", "Predicted Lineage"])

    # Display results
    st.markdown("#### View Predictions")
    update_flowchart(4)
    st.dataframe(prediction_df)

    # Download button
    csv = prediction_df.to_csv(index=False)
    st.download_button(label="Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# ---- FOOTER ----
st.markdown("""
    ---
    🔬 Developed by Bioinformatics Laboratory - College Of Computing - UM6P  
    📄 [GitHub Repository](#) | 📧 Contact: bioinformatics@um6p.ma
""")
