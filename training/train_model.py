import argparse
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(svd_model_path, sparse_matrix_path, output_model):
    """
    Loads the TruncatedSVD model, applies it to the sparse SNP matrix, 
    and trains a Random Forest classifier on the transformed features.
    
    Args:
        svd_model_path (str): Path to the saved TruncatedSVD model.
        sparse_matrix_path (str): Path to the sparse SNP matrix file (.npz).
        output_model (str): Path to save the trained Random Forest model.
    """
    print(f"Loading TruncatedSVD model from {svd_model_path}...")
    svd = joblib.load(svd_model_path)

    print(f"Loading sparse SNP matrix from {sparse_matrix_path}...")
    sparse_matrix = sp.load_npz(sparse_matrix_path)

    print("Applying TruncatedSVD transformation to SNP matrix...")
    X_transformed = svd.transform(sparse_matrix)

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_transformed, None)  # No explicit labels in this step, assuming supervised labels are not needed here.

    print(f"Saving trained model to {output_model}...")
    joblib.dump(model, output_model)

    print("✅ Model training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier on SNP features using a pre-trained TruncatedSVD model.")
    parser.add_argument("--svd_model", type=str, required=True, help="Path to the saved TruncatedSVD model.")
    parser.add_argument("--sparse_matrix", type=str, required=True, help="Path to the sparse SNP matrix file (.npz).")
    parser.add_argument("--output_model", type=str, default="models/random_forest.pkl", help="Path to save the trained Random Forest model.")

    args = parser.parse_args()

    train_random_forest(args.svd_model, args.sparse_matrix, args.output_model)
