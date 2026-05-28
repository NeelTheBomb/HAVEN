import shutil
from pathlib import Path

from utils import utils
import  subprocess
import pandas as pd
import os
from Bio import SeqIO

def run(train_df, test_df, blast_settings):
    id_col = blast_settings["id_col"]
    sequence_col = blast_settings["sequence_col"]
    label_col = blast_settings["label_col"]
    output_dir = os.path.join(blast_settings["output_dir"], "blast")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seed = blast_settings["seed"]
    n_threads = blast_settings["n_threads"]

    labels = train_df[label_col].unique().tolist()
    print(f"Unique labels: {labels}")
    #blastdbs = []
    test_fasta_filepath = utils.convert_to_fasta(test_df, [id_col, sequence_col], output_dir, f"{seed}-test")
    result_df = pd.DataFrame(test_df[id_col])
    similarity_scores_dfs = []
    for label in labels:
        # 1. construct the database
        blastdb_name = f"{seed}-train-{label}"
        train_fasta_filepath = custom_convert_to_fasta(train_df[train_df[label_col] == label], sequence_col, output_dir, f"{seed}-train-{label}")
        # create a BLAST database of the training dataset
        db_creation_output = subprocess.run(["makeblastdb", "-in", train_fasta_filepath, "-parse_seqids", "-dbtype", "prot", "-title", blastdb_name ], capture_output=True)
        print(f"\n{db_creation_output}")
        #blastdbs.append(train_fasta_filepath)

        blast_results_filepath = os.path.join(output_dir, f"{seed}-blast-results-label{str(label)}.txt")
        blast_search_output  = subprocess.run(["blastp",
                        "-db", train_fasta_filepath,
                        "-query", test_fasta_filepath,
                        "-out", blast_results_filepath,
                        "-outfmt", "10 qseqid sseqid pident evalue bitscore", # outfmt=10: Comma-separated values
                        "-max_target_seqs", "5",
                        "-num_threads", str(n_threads)],
                       capture_output=True)
        print(blast_search_output)
        df = pd.read_csv(blast_results_filepath, names=[id_col, "target_seq_id", label, "evalue", "bitscore"])
        df[label] = df[label] / 100
        similarity_scores_df = df[[id_col, "target_seq_id", label]]
        similarity_scores_df.rename(columns={label: "similarity_score"}, inplace=True)
        similarity_scores_df["target_label"] = label
        similarity_scores_df.join(test_df[[id_col, label_col]].set_index(id_col), on=id_col)
        similarity_scores_dfs.append(similarity_scores_df)
        df.drop_duplicates(subset=[id_col], keep="first", inplace=True)
        result_df = result_df.join(df[[id_col, label]].set_index(id_col), on=id_col)
    result_df.set_index(id_col, inplace=True)
    result_df = result_df.div(result_df.sum(axis=1), axis=0) # normalize identity scores between 0 to 1.
    result_df = test_df.join(result_df, on=id_col)
    # clear all temporary BLAST files
    shutil.rmtree(output_dir)
    return result_df, pd.concat(similarity_scores_dfs)


def custom_convert_to_fasta(df, seq_col, output_dir, output_filename):
    tab_file_path = os.path.join(output_dir, f"{output_filename}.tab")
    with open(tab_file_path, "w+") as f:
        df[seq_col].reset_index().to_csv(f, sep="\t", index=False, header=False)

    fasta_file_path = os.path.join(output_dir, f"{output_filename}.fasta")
    records_count = SeqIO.convert(in_file=tab_file_path, in_format="tab",
                  out_file=fasta_file_path, out_format="fasta")
    print(f"Converted {records_count} records from csv to fasta")
    return fasta_file_path