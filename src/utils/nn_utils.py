import torch.nn as nn
import torch
import copy

from training_accessories.focal_loss import FocalLoss


def create_clones(module, N):
    """
    Returns create N identical layers of a given neural network module,
    examples of modules: feed-forward, multi-head attention, or even a layer of encoder (which has multiple layers multi-head attention and feed-forward layers within it)
    :param module: neural network module
    :param N: number of clones to be created
    :return: List of N clones of module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_device(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    device = "cpu"
    if tensor is None:
        if torch.cuda.is_available():
            device = "cuda"
    else:
        if tensor.is_cuda:
            device = "cuda"
    return device



def get_criterion(loss, class_weights=None):
    criterion = nn.CrossEntropyLoss()  # default
    if loss == "MultiMarginLoss":
        criterion = nn.MultiMarginLoss()
    if loss == "FocalLoss":
#        criterion = FocalLoss(alpha=class_weights, gamma=2) ### ONLY for LOSS TESTING
        criterion = FocalLoss(alpha=class_weights, gamma=1)
    return criterion


def init_weights(module: nn.Module, initialization_type: str, bias_init_value=0):
    try:
        if initialization_type == "uniform":
            # drawn from uniform distribution between 0 and 1
            torch.nn.init.uniform_(module.weight, a=0., b=1.)
        elif initialization_type == "normal":
            # drawn from normal distribution with mean 0 and standard deviation 1
            torch.nn.init.normal_(module.weight, mean=0., std=1.)
        elif initialization_type == "zeros":
            # initialize with all zeros
            torch.nn.init.zeros_(module.weight)
        elif initialization_type == "ones":
            # initialize with all ones
            torch.nn.init.ones_(module.weight)
        else:
            print(f"ERROR: Unsupported module weight initialization type {initialization_type}")

        # initialize bias with bias_init_value
        # default bias_init_value=0
        module.bias.data.fill_(bias_init_value)
    except AttributeError as ae:
        # ignore layers which do not have the weight and/or bias attributes
        print(f"WARNING: {ae}")
        pass


def save_checkpoint(model, optimizer, lr_scheduler, epoch, file_path):
    checkpoint = {
        "model_state_dict": model,
        "optimizer_state_dict": optimizer,
        "lr_scheduler_state_dict": lr_scheduler,
        "epoch": epoch
    }
    torch.save(checkpoint, file_path.format(checkpt=epoch))


def load_model_from_checkpoint(model, file_path):
    checkpoint = torch.load(file_path, map_location=get_device())
    model_state_dict = checkpoint["model_state_dict"]

    modified_model_state_dict = {}
    for k, v in model_state_dict.items():
        if "encoder_model." in k:
            k = k.partition("encoder_model.")[-1]
            modified_model_state_dict[k] = v
    model.load_state_dict(modified_model_state_dict)
    return model

def load_checkpoint(model, optimizer, lr_scheduler, file_path):
    checkpoint = torch.load(file_path, map_location=get_device())
    last_epoch = checkpoint["epoch"]

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    lr_scheduler.last_epoch = last_epoch

    return model, optimizer, lr_scheduler, last_epoch


def set_model_grad(model, grad_value):
    for param in model.parameters():
        param.requires_grad = grad_value
    return model