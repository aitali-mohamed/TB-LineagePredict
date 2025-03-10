# TB-LineagePredict

## Overview
TB-LineagePredict is a Streamlit-based application for *Mycobacterium tuberculosis* (MTB) lineage classification using machine learning. The tool uses **Truncated Singular Value Decomposition (TruncatedSVD)** for feature extraction and a **Random Forest model** for classification.

## Features
- Upload **VCF or TXT files** for analysis.
- Extract features using **TruncatedSVD**.
- Classify MTB lineages with **Random Forest**.
- View **model predictions and confidence scores**.
- Download results for further analysis.

## Installation
### **1️⃣ Clone the repository**
```sh
git clone https://github.com/your-username/TB-LineagePredict.git
cd TB-LineagePredict
```
### **2️⃣  Create a virtual environment and install dependencies**
```sh
python -m venv env
source env/bin/activate   # Windows: env\\Scripts\\activate
pip install -r requirements.txt
```
### **3️⃣ Run the Streamlit app**
```sh
streamlit run src/tb_lineagepredict.py
```

## Training & Fine-Tuning
To retrain or fine-tune the model on a new dataset, see the training/README.md for instructions.

## Contact
📧 For issues, contact bioinformatics@um6p.ma