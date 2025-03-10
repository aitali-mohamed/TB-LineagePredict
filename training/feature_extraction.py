import argparse
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

def apply_truncated_svd(input_file_sparse, output_file, model_output, n_components=200):
    """
    Applies TruncatedSVD to the sparse SNP matrix and saves the transformed features.
    
    Args:
        input_file_sparse (str): Path to the sparse SNP matrix file (.npz).
        output_file (str): Path to save the transformed features.
        model_output (str): Path to save the trained TruncatedSVD model.
        n_components (int): Number of SVD components to retain.
    """
    print(f"Loading sparse SNP matrix from {input_file_sparse}...")
    sparse_matrix = sp.load_npz(input_file_sparse)

    print(f"Applying TruncatedSVD with {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    transformed_features = svd.fit_transform(sparse_matrix)

    print(f"Saving transformed features to {output_file}...")
    pd.DataFrame(transformed_features).to_csv(output_file, index=False)

    print(f"Saving TruncatedSVD model to {model_output}...")
    joblib.dump(svd, model_output)

    print("✅ Feature extraction completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply TruncatedSVD for feature extraction from SNP data.")
    parser.add_argument("--input_file_sparse", type=str, required=True, help="Path to the sparse SNP matrix file (.npz).")
    parser.add_argument("--output_file", type=str, default="data/snp_transformed.csv", help="Path to save transformed features.")
    parser.add_argument("--model_output", type=str, default="models/truncated_svd.pkl", help="Path to save TruncatedSVD model.")
    parser.add_argument("--n_components", type=int, default=200, help="Number of SVD components to retain.")

    args = parser.parse_args()

    apply_truncated_svd(args.input_file_sparse, args.output_file, args.model_output, args.n_components)
