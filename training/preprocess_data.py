import os
import argparse
import pandas as pd
import numpy as np
import pysam

def extract_mutations_vcf(vcf_file):
    """ Extract SNP positions from a VCF file and return a dictionary of mutations. """
    mutations = {}
    try:
        vcf = pysam.VariantFile(vcf_file)
        for record in vcf:
            mutations[record.pos] = 1  # Mutation present at this position
    except Exception as e:
        print(f"Error processing {vcf_file}: {e}")
    return mutations

def extract_mutations_txt(txt_file):
    """ Extract SNP positions from a TXT file and return a dictionary of mutations. """
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
    """ Iterates through all files in the input folder, extracts SNPs, and creates a DataFrame of mutation presence/absence. """
    data = {}

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        strain_id = filename.split(".")[0]  # Extract strain name
        
        # Extract SNP data based on file type
        if filename.endswith(".vcf") or filename.endswith(".vcf.gz"):
            mutations = extract_mutations_vcf(file_path)
        elif filename.endswith(".txt"):
            mutations = extract_mutations_txt(file_path)
        else:
            print(f"Skipping unsupported file format: {filename}")
            continue

        data[strain_id] = mutations

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index").fillna(0).astype(int)

    # Ensure consistent ordering of SNP positions
    df = df.reindex(sorted(df.columns), axis=1)

    return df

def save_snp_matrix(input_folder, output_file):
    """ Runs the full preprocessing pipeline and saves the SNP matrix to a CSV file. """
    print(f"Processing files from {input_folder} and extracting SNPs...")
    snp_matrix = process_files(input_folder)

    print(f"Saving SNP matrix to {output_file}...")
    snp_matrix.to_csv(output_file)
    print("✅ SNP matrix saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw VCF/TXT files into a SNP matrix.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the directory containing VCF/TXT files.")
    parser.add_argument("--output_file", type=str, default="data/snp_matrix.csv", help="Output file path for the SNP matrix.")

    args = parser.parse_args()

    save_snp_matrix(args.input_folder, args.output_file)
