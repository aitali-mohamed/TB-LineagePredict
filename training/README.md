# Model Training for TB-LineagePredict

## Overview
This folder contains scripts to **train and evaluate the Random Forest model** used in the TB-LineagePredict application. The training pipeline includes:
1. **Preprocessing raw genomic data** into a sparse SNP matrix.
2. **Applying TruncatedSVD** for feature extraction.
3. **Training a Random Forest classifier** on transformed features.
4. **Evaluating the model using classification metrics and phylogenetic validation.**

## Training Pipeline

### **1️⃣ Preprocess Raw Data**
Convert raw **VCF/TXT** files into a **sparse SNP matrix**:
```sh
python training/preprocess_data.py --input_folder data/raw_vcf --output_file_sparse data/snp_matrix.npz
```
### **2️⃣ Apply Feature Extraction (TruncatedSVD)**
Extract features from the SNP matrix using TruncatedSVD:
```sh
python training/feature_extraction.py --input_file_sparse data/snp_matrix.npz --output_file data/snp_transformed.csv --model_output models/truncated_svd.pkl --n_components 200
```
### **3️⃣ Train the Model**
Train the Random Forest model on TruncatedSVD-transformed features:
```sh
python training/train_model.py --svd_model models/truncated_svd.pkl --sparse_matrix data/snp_matrix.npz --output_model models/random_forest.pkl
```
### **4️⃣ Evaluate the Model**
Compute classification metrics (Precision, Recall, F1-score, ARI, NMI):
```sh
python training/evaluate_model.py --rf_model models/random_forest.pkl --svd_model models/truncated_svd.pkl --sparse_matrix data/snp_matrix.npz --test_labels data/labels_test.csv
```
## Customizing the Training Process
Modify the --n_components parameter in feature_extraction.py to change the number of SVD components.
Adjust hyperparameters in train_model.py to fine-tune the Random Forest classifier.
Use custom datasets by specifying --input_folder in preprocess_data.py.
## Contact
For issues related to model training, contact bioinformatics@um6p.ma