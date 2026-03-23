import utils

def run(train_df, test_df, blast_settings):
    id_col = blast_settings["id_col"]
    sequence_col = blast_settings["sequence_col"]
    output_dir = blast_settings["output_dir"]
    seed = blast_settings["seed"]

    # 1. construct the database
    train_fasta_filepath = utils.convert_to_fasta(train_df, [id_col, sequence_col], output_dir,
                                                  str(input_split_seeds[iter]))
