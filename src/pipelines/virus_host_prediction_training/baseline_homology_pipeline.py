import os
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from utils import utils, dataset_utils, kmer_utils, visualization_utils
from models.baseline.homology import blast_search


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]
    input_split_seeds = input_settings["split_seeds"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    # classification settings
    classification_settings = config["classification_settings"]
    models = classification_settings["models"]
    label_settings = classification_settings["label_settings"]
    sequence_settings = classification_settings["sequence_settings"]
    n_iters = classification_settings["n_iterations"]
    classification_type = classification_settings["type"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    virus_col = sequence_settings["virus_col"]
    label_col = label_settings["label_col"]

    # wandb_config = {
    #     "n_epochs": training_settings["n_epochs"],
    #     "lr": training_settings["max_lr"],
    #     "max_sequence_length": sequence_settings["max_sequence_length"],
    #     "dataset": input_file_names[0]
    # }
    output_filename_prefix = f"{label_col}_{classification_type}_{output_prefix}"
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    # create output_results_dir path, if it does not already exist
    Path(output_results_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    feature_importance = {}
    validation_scores = {}
    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col, virus_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=classification_type)
        # 3. Split dataset
        if classification_settings["split_input_col"]:
            train_df, test_df = dataset_utils.split_dataset_based_on_column(df, input_split_seeds[iter], classification_settings["train_proportion"],
                                                                            split_input_col=classification_settings["split_input_col"],
                                                                            label_col=label_col)
        else:
            train_df, test_df = dataset_utils.split_dataset_stratified(df, input_split_seeds[iter],
                                                                   classification_settings["train_proportion"], stratify_col=label_col)


        # 5. Perform classification
        for model in models:
            if model["active"] is False:
                print(f"Skipping {model['name']} ...")
                continue
            model_name = model["name"]
            if model_name not in results:
                # first iteration
                results[model_name] = []


            # Set necessary values within model_params object for cleaner code and to avoid passing multiple arguments.
            model["label_col"] = label_col
            model["seed"] = str(input_split_seeds[iter])
            model["id_col"] = id_col
            model["sequence_col"] = sequence_col
            model["output_dir"] = output_results_dir

            if "blast" in model_name:
                print("Executing Homology-based classification using BLAST ...")
                result_df = blast_search.run(train_df, test_df, model)
            else:
                continue

            #  Create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df[label_col]
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter
            results[model_name].append(result_df)

    # write the raw results in csv files
    utils.write_output(results, output_results_dir, output_filename_prefix, "output")