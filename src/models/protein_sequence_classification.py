from abc import abstractmethod
import torch.nn as nn
from utils import nn_utils, constants
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d


class ProteinSequenceClassification(nn.Module):
    """
    Template base for a virus host classification model_params based on deep learning methods
    Contains the MLP block for the multiclass classification
    """

    def __init__(self, input_dim=512, hidden_dim=2048, n_mlp_layers=2, n_classes=1, batch_norm=False):
        super(ProteinSequenceClassification, self).__init__()
        self.batch_norm = batch_norm

        ## Multiclass classification block
        # first linear layer: input_dim --> hidden_dim
        self.linear_ip = nn.Linear(input_dim, hidden_dim)

        # intermediate hidden layers (number = N): hidden_dim --> hidden_dim
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.linear_hidden_n = nn_utils.create_clones(self.linear_hidden, n_mlp_layers)

        # last linear layer: hidden_dim--> n_classes
        self.linear_op = nn.Linear(hidden_dim, n_classes)

        if self.batch_norm:
            self.batch_norm_ip = BatchNorm1d(hidden_dim)
            self.batch_norm_hidden = BatchNorm1d(hidden_dim)
            self.batch_norm_hidden_n = nn_utils.create_clones(self.batch_norm_hidden, n_mlp_layers)

    @abstractmethod
    def get_embedding(self, X):
        """
        Method to generate the embedding for a given input batch of protein sequences.
        This method is called from within the forward()
        Keep this method explicit and separate for downstream interpretability analysis
        so that the embedding for a given input can be retrieved without having to do a forward pass on the input.

        Args:
            X: input batch of sequences as a tensor
        Return:
            embedding for X
        """
        raise NotImplementedError

    def forward(self, X, embedding_only=False):
        """
        Method to do the forward pass of the input batch of sequences.
        Template implementation:
        1. Do any model_params specific pre-processing of input sequences, if required
        2. Invoke get_embedding()
        3. Do any model_params specific post-processing of embeddings, if required
        4. if embedding_only, return the embedding
        5. Invoke forward_classification_block()

        Args:
            X: input batch of sequences as a tensor
            embedding_only: boolean
        Return:
            output logits for n_classes
        """
        self.input_embedding = self.get_embedding(X)

        if embedding_only:
            # used in Few Shot Learning
            # Hack to use DataParallel and run on multiple GPUs since we can only call __call__() --> forward() using DataParallel
            return self.input_embedding, self.get_classification_block_embedding(self.input_embedding)

        return self.forward_classification_block(self.input_embedding)

    def forward_classification_block(self, X):
        """
        Method to do the forward pass through the multiclass classification block with the embeddings passed as input.

        Args:
            X: embeddings of input batch of sequences
        Return:
            output logits for n_classes
        """
        batch_size = X.shape[0]  # batch_size

        # input linear layer
        X = F.relu(self.linear_ip(X))
        if self.batch_norm and batch_size > 1:  # batch_norm is applicable only when batch_size is > 1
            X = self.batch_norm_ip(X)
        # hidden
        for i, linear_layer in enumerate(self.linear_hidden_n):
            X = F.relu(linear_layer(X))
            if self.batch_norm and batch_size > 1:  # batch_norm is applicable only when batch_size is > 1
                X = self.batch_norm_hidden_n[i](X)

        y = self.linear_op(X)
        return y

    def get_classification_block_embedding(self, X):
        """
        Method to do the forward pass through the multiclass classification block with the embeddings passed as input.

        Args:
            X: embeddings of input batch of sequences
        Return:
            output logits for n_classes
        """
        batch_size = X.shape[0]  # batch_size

        # input linear layer
        X = F.relu(self.linear_ip(X))
        if self.batch_norm and batch_size > 1:  # batch_norm is applicable only when batch_size is > 1
            X = self.batch_norm_ip(X)
        # hidden
        for i, linear_layer in enumerate(self.linear_hidden_n):
            X = F.relu(linear_layer(X))
            if self.batch_norm and batch_size > 1:  # batch_norm is applicable only when batch_size is > 1
                X = self.batch_norm_hidden_n[i](X)
        return X

    @staticmethod
    @abstractmethod
    def get_model(model_params):
        """
        Static method to instantiate an object of the virus host prediction class with the model_params.
        Option data_parallel will allow for parallelization of the execution
        Args:
            model_params: dict with all the model_params params required to instantiate the virus host prediction class
            data_parallel: enable parallelization of execution on multiple GPUs using DataParallel
        """
        raise NotImplementedError

    @staticmethod
    def return_model(model, data_parallel):
        """
        Return the model_params based on whether parallelized execution on multiple GPUs is required.
        Making it static as the object of the class is an argument itself. If not static, it will lead to circular dependency.
        Args:
            model: model_params to be returned
            data_parallel: boolean
        Return: model_params converted to data_parallel mode if required
        """
        if data_parallel:
            # Capability to distribute data for parallelization
            return nn.DataParallel(model.to(nn_utils.get_device()))
        else:
            return model.to(nn_utils.get_device())