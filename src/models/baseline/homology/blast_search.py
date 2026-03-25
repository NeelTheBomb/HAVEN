from utils import utils
import  subprocess
import pandas as pd
import os

def run(train_df, test_df, blast_settings):
    id_col = blast_settings["id_col"]
    sequence_col = blast_settings["sequence_col"]
    label_col = blast_settings["label_col"]
    output_dir = os.path.join(blast_settings["output_dir"], "blast")
    seed = blast_settings["seed"]

    labels = train_df[label_col].unique().tolist()
    print(f"Unique labels: {labels}")
    blastdbs = []
    for label in labels:
        # 1. construct the database
        blastdb_name = f"{seed}-train-{label}"
        train_fasta_filepath = utils.convert_to_fasta(train_df[train_df[label_col] == label], [id_col, sequence_col], output_dir, f"{seed}-train-{label}")
        # create a BLAST database of the training dataset
        db_creation_output = subprocess.run(["makeblastdb", "-in", train_fasta_filepath, "-parse_seqids", "-dbtype", "prot", "-title", blastdb_name ], capture_output=True, shell=True)
        print(f"\n{db_creation_output}")
        blastdbs.append(train_fasta_filepath)

    test_fasta_filepath = utils.convert_to_fasta(test_df[:10], [id_col, sequence_col], output_dir, f"{seed}-test")
    blast_results_filepath = os.path.join(output_dir, f"{seed}-blast-results.txt")
    x  = subprocess.run(["blastp",
                    "-db", train_fasta_filepath,
                    "-query", test_fasta_filepath,
                    "-out", blast_results_filepath,
                    "-outfmt", "10 qseqid sseqid pident length mismatch gapopen evalue bitscore"],
                   capture_output=True, shell=True) # outfmt=10: Comma-separated values
    print(x)
    print("BLAST search completed")