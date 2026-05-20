"""
Common utility functions to train, validate, and test deep learning models for the virus host prediction task
"""
import wandb
import pandas as pd
import tqdm
from statistics import mean
import torch
import torch.nn.functional as F

from utils import nn_utils


def run_epoch(model, train_dataset_loader, val_dataset_loader, criterion,
              optimizer, lr_scheduler, early_stopper, model_id, epoch):
    # training
    model.train()
    for _, record in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
        input, label = record

        optimizer.zero_grad()

        output = model(input)
        output = output.to(nn_utils.get_device())

        loss = criterion(output, label.long())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        model.train_iter += 1
        curr_lr = lr_scheduler.get_last_lr()[0]
        train_loss = loss.item()
        wandb.log({
            "learning-rate": float(curr_lr),
            "training-loss": float(train_loss)
        })
        pbar.set_description(
            f"{model_id}/training-loss = {float(train_loss)}, model.n_iter={model.train_iter}, epoch={epoch + 1}")

    # validation
    val_loss = validate_model(model, val_dataset_loader, criterion, model_id, epoch)
    early_stopper(model, val_loss)
    return model


def validate_model(model, dataset_loader, criterion, model_id, epoch):
    with torch.no_grad():
        model.eval()

        val_loss = []
        for _, record in enumerate(pbar := tqdm.tqdm(dataset_loader)):
            input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            loss = criterion(output, label.long())
            curr_val_loss = loss.item()
            model.val_iter += 1

            # log validation loss
            wandb.log({
                "validation-loss": float(curr_val_loss)
            })
            pbar.set_description(
                f"{model_id}/validation-loss = {float(curr_val_loss)}, model.n_iter={model.val_iter}, epoch={epoch + 1}")
            val_loss.append(curr_val_loss)

    return mean(val_loss)


def test_model(model, dataset_loader):
    with torch.no_grad():
        model.eval()

        results = []
        for _, record in enumerate(pbar := tqdm.tqdm(dataset_loader)):
            input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True)


def test_model_analysis(model, dataset_loader, id_col):
    with torch.no_grad():
        model.eval()

        results = []
        for _, record in enumerate(pbar := tqdm.tqdm(dataset_loader)):
            id, input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df[id_col] = id
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True)