#!/usr/src/env python
import os
import argparse
import pandas as pd
from pathlib import Path
import subprocess



def parse_args():
    parser = argparse.ArgumentParser(description='Generate perturbed sequences for a given dataset')
    parser.add_argument("-if", "--input_file", required=True,
                        help="File with input sequences.\n")
    parser.add_argument("-id", "--id_col", required=True,
                        help="Name of the column with the sequence id.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")

    args = parser.parse_args()
    return args

def download_proteome(id, output_dir):
    print(f"Downloading proteome {id}")
    try:
        output_file_name = f"{id}.zip"
        subprocess.run(["datasets", "download", "virus", "genome", "accession", f"{id}.1", "--include", "protein", "--filename", output_file_name])
        os.replace(output_file_name, os.path.join(output_dir, output_file_name))

    except Exception as e:
        print(f"Failed to download proteome {id}")
        print(e)


def download_proteomes(input_file, id_col, output_dir):
    df = pd.read_csv(input_file)
    print(f"Dataset size: {df.shape}")
    ids = list(df[id_col].unique())
    print("Number of unique ids: ", len(ids))

    Path(os.path.dirname(output_dir)).mkdir(parents=True, exist_ok=True)

    for id in ids:
        download_proteome(id, output_dir)
    return

def main():
    config = parse_args()
    download_proteomes(config.input_file, config.id_col, config.output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)