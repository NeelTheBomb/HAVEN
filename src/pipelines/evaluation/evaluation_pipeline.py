import os
import pandas as pd
from pathlib import Path
from evaluation.binary_class_evaluation import BinaryClassEvaluation
from evaluation.multi_class_evaluation import MultiClassEvaluation


def execute(config):
    input_settings = config["input_settings"]
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_evaluation_dir = output_settings["evaluation_dir"]
    output_visualization_dir = output_settings["visualization_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    label_mappings = config["label_mappings"]

    evaluation_settings = config["evaluation_settings"]
    evaluation_type = evaluation_settings["type"]

    df = read_inputs(input_settings, label_mappings=label_mappings)
    output_file_name = get_output_file_name(output_settings)
    evaluation_output_file_base_path = os.path.join(output_dir, output_evaluation_dir, output_dataset_dir)
    visualization_output_file_base_path = os.path.join(output_dir, output_visualization_dir, output_dataset_dir)
    # create any missing parent directories
    Path(os.path.dirname(evaluation_output_file_base_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(visualization_output_file_base_path)).mkdir(parents=True, exist_ok=True)

    if evaluation_type == "binary":
        evaluation_executor = BinaryClassEvaluation(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name)
    elif evaluation_type == "multi":
        evaluation_executor = MultiClassEvaluation(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name, label_mappings)
    else:
        print(f"ERROR: Unsupported type of evaluation: evaluation_settings.type = {evaluation_type}")

    evaluation_executor.execute()
    return


def read_inputs(input_settings, label_mappings=None):
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]

    df = None
    inputs = []
    for key, file_name in input_file_names.items():
        input_file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(input_file_path, index_col=0)
        print(f"input file = {input_file_path} --> results size = {df.shape}")
        if label_mappings:
            df.rename(columns=label_mappings, inplace=True)
            df["y_true"] = df["y_true"].replace(label_mappings)
        df["experiment"] = key
        inputs.append(df)
    df = pd.concat(inputs)
    return df


def get_output_file_name(output_settings):
    output_prefix = output_settings["prefix"]
    output_prefix = "evaluation" if output_prefix is None else output_prefix
    output_file_name = output_prefix

    return output_file_name
