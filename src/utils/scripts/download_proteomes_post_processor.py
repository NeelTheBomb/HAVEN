#!/usr/src/env python
import os
import argparse
import pandas as pd
from pathlib import Path
import shutil
from zipfile import ZipFile
from Bio import SeqIO
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Generate perturbed sequences for a given dataset')
    parser.add_argument("-if", "--input_file", required=True,
                        help="File with input sequences.\n")
    parser.add_argument("-id", "--id_col", required=True,
                        help="Name of the column with the sequence id.\n")
    parser.add_argument("-pd", "--proteome_dir", required=True,
                        help="Absolute path to proteome directory.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    parser.add_argument("--fasta_to_csv", action="store_true",
                        help="Convert the downloaded fasta files to csv format.\n")

    args = parser.parse_args()
    return args


def process_proteomes(input_file, id_col, proteome_dir, output_dir):
    df = pd.read_csv(input_file)
    print(f"Dataset size: {df.shape}")
    ids = list(df[id_col].unique())
    print("Number of unique ids: ", len(ids))

    Path(os.path.dirname(output_dir)).mkdir(parents=True, exist_ok=True)

    proteome_not_present_count = 0
    for id in ids:
        print("Processing", id)
        try:
            input_zipfile_name = f"{id}.zip"
            proteome_zip_file_path = os.path.join(proteome_dir, input_zipfile_name)
            if os.path.exists(proteome_zip_file_path):
                id_dir = os.path.join(proteome_dir, id)
                Path(os.path.dirname(id_dir)).mkdir(parents=True, exist_ok=True)
                with ZipFile(proteome_zip_file_path, "r") as zip_object:
                    zip_object.extract("ncbi_dataset/data/protein.faa", path=id_dir)

                zip_object.close()
                shutil.move(os.path.join(id_dir, "ncbi_dataset/data/protein.faa"), os.path.join(output_dir, f"{id}.faa"))
                shutil.rmtree(id_dir)
            else:
                proteome_not_present_count += 1
        except Exception as e:
            print(e)


    print(f"Proteomes not present count: {proteome_not_present_count}")
    return

def convert_fasta_to_csv(id, fasta_file_path):
    proteins = []
    with open(fasta_file_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            proteins.append({
                "proteome_id": id,
                "protein_id": record.id.split(":")[0],
                "sequence": str(record.seq),
                "protein": record.description.split(" [")[0][len(record.id)+1:]
            })
    return proteins

def process_fasta_files(input_file, id_col, proteome_dir, output_dir):
    df = pd.read_csv(input_file)
    print(f"Dataset size: {df.shape}")
    ids = list(df[id_col].unique())
    print("Number of unique ids: ", len(ids))
    no_proteome_count = 0
    processed_proteome_count = 0
    protein_sequences = []
    for id in ids:
        fasta_file_path = os.path.join(proteome_dir, f"{id}.faa")
        if os.path.exists(fasta_file_path):
            proteins = convert_fasta_to_csv(id, fasta_file_path)
            print(f"{id}: {len(proteins)} proteins")
            protein_sequences.extend(proteins)
            processed_proteome_count += 1
        else:
            no_proteome_count += 1
    print(f"processed_proteome_count: {processed_proteome_count}")
    print(f"no_proteome_count: {no_proteome_count}")
    print(f"Total number of sequences = {len(protein_sequences)}")
    pd.DataFrame(protein_sequences).to_csv(os.path.join(output_dir, Path(input_file).stem + "_proteome_proteins.csv"), index=False)

def main():
    config = parse_args()
    if config.fasta_to_csv:
        process_fasta_files(config.input_file, config.id_col,  config.proteome_dir, config.output_dir)
    else:
        process_proteomes(config.input_file, config.id_col,  config.proteome_dir, config.output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)