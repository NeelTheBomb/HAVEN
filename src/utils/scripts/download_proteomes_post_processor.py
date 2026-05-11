#!/usr/src/env python
import os
import argparse
import pandas as pd
from pathlib import Path
import subprocess
import shutil
from zipfile import ZipFile

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


    print(f"Proteomes not present count: {proteome_not_present_count}")
    return

def main():
    config = parse_args()
    process_proteomes(config.input_file, config.id_col,  config.proteome_dir, config.output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)