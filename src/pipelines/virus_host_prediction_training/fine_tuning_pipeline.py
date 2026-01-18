import os
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
import torch
import wandb

from utils import utils, dataset_utils, nn_utils, constants, mapper, training_utils
from training_accessories.early_stopping import EarlyStopping
from models.baseline.nlp.transformer.transformer import TransformerEncoder


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

    sequence_settings = config["sequence_settings"]
    pre_train_settings = config["pre_train_settings"]

    fine_tune_settings = config["fine_tune_settings"]
    label_settings = fine_tune_settings["label_settings"]
    training_settings = fine_tune_settings["training_settings"]

    pre_train_encoder_settings = pre_train_settings["encoder_settings"]
    pre_train_encoder_settings["vocab_size"] = constants.VOCAB_SIZE
    n_iters = fine_tune_settings["n_iterations"]

    sequence_settings["max_sequence_length"] = pre_train_encoder_settings["max_seq_len"]

    tasks = fine_tune_settings["task_settings"]
    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]
    results = {}

    wandb_config = {
        "n_epochs_freeze": training_settings["n_epochs_freeze"],
        "n_epochs_unfreeze": training_settings["n_epochs_unfreeze"],
        "lr": training_settings["max_lr"],
        "max_sequence_length": sequence_settings["max_sequence_length"],
        "dataset": input_file_names[0],
        "output_prefix": output_prefix
    }

    # fine_tune_model store filepath
    fine_tune_model_filepath = os.path.join(output_dir, results_dir, sub_dir, "{output_prefix}_{task_id}_itr{itr}.pth")
    Path(os.path.dirname(fine_tune_model_filepath)).mkdir(parents=True, exist_ok=True)

    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                cols=[id_col, sequence_col, label_col])
        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                           classification_type=fine_tune_settings["classification_type"])

        train_dataset_loader = None
        val_dataset_loader = None
        test_dataset_loader = None
        # 3. Split dataset
        if fine_tune_settings["split_input"]:
            # full df into training and testing datasets in the ratio configured in the config file
            train_df, test_df = dataset_utils.split_dataset_stratified(df, input_settings["split_seeds"][iter],
                                                                       fine_tune_settings["train_proportion"], stratify_col=label_col)
            # split testing set into validation and testing datasets in equal proportion
            # so 80:20 will now be 80:10:10
            val_df, test_df = dataset_utils.split_dataset_stratified(test_df, input_split_seeds[iter], 0.5, stratify_col=label_col)
            train_dataset_loader = dataset_utils.get_dataset_loader(train_df, sequence_settings, label_col)
            val_dataset_loader = dataset_utils.get_dataset_loader(val_df, sequence_settings, label_col)
            test_dataset_loader = dataset_utils.get_dataset_loader(test_df, sequence_settings, label_col)
        else:
            # used in zero shot evaluation, where split_input=False in fine_tune_settings and mode=test in task
            test_dataset_loader = dataset_utils.get_dataset_loader(df, sequence_settings, label_col)

        fine_tune_model = None
        for task in tasks:
            task_id = task["id"] # unique identifier
            task_name = task["name"]
            mode = task["mode"]

            if task["active"] is False:
                print(f"Skipping {task_name} ...")
                continue

            # load pre-trained encoder model_params
            pre_trained_encoder_model = TransformerEncoder.get_transformer_encoder(pre_train_encoder_settings, task["cls_token"])
            pre_trained_model_path = pre_train_settings["model_path"]
            if pre_trained_model_path:
                pre_trained_encoder_model.load_state_dict(
                    torch.load(pre_trained_model_path, map_location=nn_utils.get_device()))

            # HACK to load models from checkpoints. CAUTION: Use only under dire circumstances
            # pre_trained_encoder_model = nn_utils.load_model_from_checkpoint(pre_trained_encoder_model,
            #                                                                pre_train_settings["model_path"])

            # set the pre_trained model_params within the task config
            task["pre_trained_model"] = pre_trained_encoder_model

            # add maximum sequence length of pretrained model_params as the segment size from the sequence_settings
            # in pre_train_encoder_settings it has been incremented by 1 to account for CLS token
            task["segment_len"] = sequence_settings["max_sequence_length"]

            if task_name in mapper.model_map:
                print(f"Executing {task_name} in {mode} mode.")
                fine_tune_model = mapper.model_map[task_name].get_model(model_params=task)
            else:
                print(f"ERROR: Unknown model {task_name}.")
                continue

            if task_id not in results:
                # first iteration
                results[task_id] = []

            # Initialize Weights & Biases for each run
            wandb_config["hidden_dim"] = task["hidden_dim"]
            wandb_config["n_mlp_layers"] = task["n_mlp_layers"]

            wandb.init(project="haven",
                       config=wandb_config,
                       group=fine_tune_settings["experiment"],
                       job_type=task_id,
                       name=f"iter_{iter}")

            if mode == "train":
                # retraining the model_params for the fine_tuning task
                result_df, fine_tune_model = run_task(fine_tune_model, train_dataset_loader, val_dataset_loader, test_dataset_loader,
                                                   task["loss"], training_settings, task_id)
            elif mode == "test":
                # used for zero-shot evaluation
                # load the pre-trained and fine_tuned model_params
                fine_tune_model.load_state_dict(torch.load(task["fine_tuned_model_path"]))
                result_df = training_utils.test_model(fine_tune_model, test_dataset_loader)
            else:
                print(f"ERROR: Unsupported mode '{mode}'. Supported values: 'train', 'test'.")
                exit(1)

            #  create the result dataframe and remap the class indices to original input labels
            result_df.rename(columns=index_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(index_label_map)
            result_df["itr"] = iter
            results[task_id].append(result_df)

            if fine_tune_settings["save_model"]:
                # save the fine_tuned model_params
                model_filepath = fine_tune_model_filepath.format(output_prefix=output_prefix, task_id=task_id, itr=iter)
                torch.save(fine_tune_model.state_dict(), model_filepath)
                print(f"Model output written to {model_filepath}")

            wandb.finish()

    # write the raw results in csv files
    output_results_dir = os.path.join(output_dir, results_dir, sub_dir)
    utils.write_output(results, output_results_dir, output_prefix, "output")


def run_task(model, train_dataset_loader, val_dataset_loader, test_dataset_loader, loss, training_settings, task_id):
    class_weights = utils.get_class_weights(train_dataset_loader).to(nn_utils.get_device())
    criterion = nn_utils.get_criterion(loss, class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    n_epochs_freeze = training_settings["n_epochs_freeze"]
    n_epochs_unfreeze = training_settings["n_epochs_unfreeze"]
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=float(training_settings["max_lr"]),
        epochs=n_epochs_freeze + n_epochs_unfreeze,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=training_settings["pct_start"],
        anneal_strategy='cos',
        div_factor=training_settings["div_factor"],
        final_div_factor=training_settings["final_div_factor"])
    early_stopper = EarlyStopping(patience=3, min_delta=0)
    model.train_iter = 0
    model.val_iter = 0

    # START: Model training with early stopping using validation
    # freeze the pretrained model_params for the first n_epochs_freeze
    # nn_utils.set_model_grad(model.module.pre_trained_model, grad_value=False)
    if isinstance(model, torch.nn.DataParallel):
        nn_utils.set_model_grad(model.module.pre_trained_model, grad_value=False)
    else:
        nn_utils.set_model_grad(model.pre_trained_model, grad_value=False)


    # train for n_epochs_freeze
    for e in range(n_epochs_freeze):
        model = training_utils.run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer,
                                         lr_scheduler, early_stopper, task_id, e)
        # check if early stopping condition was satisfied and stop accordingly
        if early_stopper.early_stop:
            print("Breaking off frozen training loop due to early stop")
            break

    # unfreeze the pretrained model_params for the next n_epochs_unfreeze
    # nn_utils.set_model_grad(model.module.pre_trained_model, grad_value=True)

    if isinstance(model, torch.nn.DataParallel):
        nn_utils.set_model_grad(model.module.pre_trained_model, grad_value=True)
    else:
        nn_utils.set_model_grad(model.pre_trained_model, grad_value=True)


    # reset early stopper
    early_stopper.reset()

    for e in range(n_epochs_unfreeze):
        model = training_utils.run_epoch(model, train_dataset_loader, val_dataset_loader, criterion, optimizer,
                                         lr_scheduler, early_stopper, task_id, e)
        # check if early stopping condition was satisfied and stop accordingly
        if early_stopper.early_stop:
            print("Breaking off unfrozen training loop due to early stop")
            break
    # END: Model training with early stopping using validation

    # choose the model_params with the lowest validation loss from the early stopper
    best_performing_model = early_stopper.get_current_best_model()

    # test the model_params
    result_df = training_utils.test_model(best_performing_model, test_dataset_loader)

    return result_df, best_performing_model