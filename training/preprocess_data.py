import os
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pysam

def extract_mutations_vcf(vcf_file):
    """Extract SNP positions from a VCF file and return a dictionary of mutations."""
    mutations = {}
    try:
        vcf = pysam.VariantFile(vcf_file)
        for record in vcf:
            mutations[record.pos] = 1  # Mutation present at this position
    except Exception as e:
        print(f"Error processing {vcf_file}: {e}")
    return mutations

def extract_mutations_txt(txt_file):
    """Extract SNP positions from a TXT file and return a dictionary of mutations."""
    mutations = {}
    with open(txt_file, "r") as file:
        for line in file:
            if line.startswith("CHROM"):  # Skip header
                continue
            parts = line.strip().split("\t")
            if len(parts) > 1:
                pos = int(parts[1])
                mutations[pos] = 1  # Mutation present at this position
    return mutations

def process_files(input_folder):
    """Iterates through all files in the input folder, extracts SNPs, and creates a sparse matrix."""
    data = {}

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        strain_id = filename.split(".")[0]  # Extract strain name
        
        if filename.endswith(".vcf") or filename.endswith(".vcf.gz"):
            mutations = extract_mutations_vcf(file_path)
        elif filename.endswith(".txt"):
            mutations = extract_mutations_txt(file_path)
        else:
            print(f"Skipping unsupported file format: {filename}")
            continue

        data[strain_id] = mutations

    df = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)
    df = df.reindex(sorted(df.columns), axis=1)  # Ensure SNPs are consistently ordered

    sparse_matrix = sp.csr_matrix(df.values)  # Convert to sparse format
    return sparse_matrix

def save_snp_matrix(input_folder, output_file_sparse):
    """Runs the full preprocessing pipeline and saves the SNP matrix as a sparse file."""
    print(f"Processing files from {input_folder} and extracting SNPs...")
    sparse_matrix = process_files(input_folder)

    print(f"Saving SNP matrix (sparse) to {output_file_sparse}...")
    sp.save_npz(output_file_sparse, sparse_matrix)

    print("✅ SNP matrix saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw VCF/TXT files into a sparse SNP matrix.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the directory containing VCF/TXT files.")
    parser.add_argument("--output_file_sparse", type=str, default="data/snp_matrix.npz", help="Output file path for the SNP matrix (Sparse format).")

    args = parser.parse_args()
    save_snp_matrix(args.input_folder, args.output_file_sparse)
