import argparse
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.metrics import classification_report, adjusted_rand_score, normalized_mutual_info_score

def evaluate_model(rf_model_path, svd_model_path, sparse_matrix_path, test_labels_path):
    """
    Loads the trained Random Forest model and TruncatedSVD, applies SVD to the sparse SNP matrix,
    and evaluates the model using classification metrics.
    
    Args:
        rf_model_path (str): Path to the trained Random Forest model.
        svd_model_path (str): Path to the saved TruncatedSVD model.
        sparse_matrix_path (str): Path to the sparse SNP matrix file (.npz).
        test_labels_path (str): Path to the test labels file (.csv).
    """
    print(f"Loading trained Random Forest model from {rf_model_path}...")
    rf_model = joblib.load(rf_model_path)

    print(f"Loading TruncatedSVD model from {svd_model_path}...")
    svd_model = joblib.load(svd_model_path)

    print(f"Loading sparse SNP matrix from {sparse_matrix_path}...")
    sparse_matrix = sp.load_npz(sparse_matrix_path)

    print("Applying TruncatedSVD transformation to test SNP matrix...")
    X_test_transformed = svd_model.transform(sparse_matrix)

    print(f"Loading test labels from {test_labels_path}...")
    y_test = pd.read_csv(test_labels_path).values.ravel()

    print("Making predictions...")
    y_pred = rf_model.predict(X_test_transformed)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained Random Forest classifier.")
    parser.add_argument("--rf_model", type=str, required=True, help="Path to the trained Random Forest model.")
    parser.add_argument("--svd_model", type=str, required=True, help="Path to the saved TruncatedSVD model.")
    parser.add_argument("--sparse_matrix", type=str, required=True, help="Path to the sparse SNP matrix file (.npz).")
    parser.add_argument("--test_labels", type=str, required=True, help="Path to the test labels file (.csv).")

    args = parser.parse_args()

    evaluate_model(args.rf_model, args.svd_model, args.sparse_matrix, args.test_labels)
