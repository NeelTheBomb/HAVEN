from utils import nn_utils
from datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID
import torch
import re


class ProteinSequenceProstT5Dataset(ProteinSequenceDatasetWithID):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col, include_id_col):
        super(ProteinSequenceProstT5Dataset, self).__init__(df, id_col, sequence_col, max_seq_len, truncate, label_col)
        self.include_id_col = include_id_col

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        sequence_length = len(sequence)

        sequence = re.sub(r"[UZOB]", "X", sequence)
        # Add spaces between each amino acid for PT5 to correctly use them
        sequence = " ".join(sequence)
        # AAs to 3Di (or if you want to embed AAs): prepend "<AA2fold>"
        sequence = "<AA2fold>" + " " + sequence

        # return a tuple of (sequence, sequence_length) as the sequence length will be used in the get_embedding() method to retrieve the embeddings of
        # only the amino acid tokens while excluding the padding, start, and end special tokens.
        if self.include_id_col:
            return record[self.id_col], (sequence, sequence_length), torch.tensor(record[self.label_col], device=nn_utils.get_device())
        else:
            return (sequence, sequence_length), torch.tensor(record[self.label_col], device=nn_utils.get_device())


class ProteinSequenceProtT5Dataset(ProteinSequenceDatasetWithID):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col, include_id_col, include_label_col=False):
        super(ProteinSequenceProtT5Dataset, self).__init__(df, id_col, sequence_col, max_seq_len, truncate, label_col)
        self.include_id_col = include_id_col
        self.include_label_col = include_label_col

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        sequence_length = len(sequence)

        sequence = re.sub(r"[UZOB]", "X", sequence)
        # Add spaces between each amino acid for PT5 to correctly use them
        sequence = " ".join(sequence)

        # return a tuple of (sequence, sequence_length) as the sequence length will be used in the get_embedding() method to retrieve the embeddings of
        # only the amino acid tokens while excluding the padding, start, and end special tokens.
        if self.include_id_col:
            return record[self.id_col], (sequence, sequence_length), torch.tensor(record[self.label_col], device=nn_utils.get_device())
        elif self.include_label_col:
            return (sequence, sequence_length), record[self.label_col]
        else:
            return (sequence, sequence_length), torch.tensor(record[self.label_col], device=nn_utils.get_device())


class ProteinSequenceESM2Dataset(ProteinSequenceDatasetWithID):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col, include_id_col):
        super(ProteinSequenceESM2Dataset, self).__init__(df, id_col, sequence_col, max_seq_len, truncate, label_col)
        self.include_id_col = include_id_col

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        sequence = sequence.replace("J", "X") # replacing J (leucine or isoleucine) with X (unknown or atypical AA)

        # format the sequence as a tuple with the id as the identifier
        formatted_sequence = (record[self.id_col], sequence)

        if self.include_id_col:
            return record[self.id_col], formatted_sequence, torch.tensor(record[self.label_col], device=nn_utils.get_device())
        else:
            return formatted_sequence, torch.tensor(record[self.label_col], device=nn_utils.get_device())


class ProteinSequenceLucaVirusDataset(ProteinSequenceDatasetWithID):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col, include_id_col):
        super(ProteinSequenceLucaVirusDataset, self).__init__(df, id_col, sequence_col, max_seq_len, truncate, label_col)
        self.include_id_col = include_id_col

    def __getitem__(self, idx: int):
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        sequence = sequence.replace("J", "X")

        formatted_sequence = (record[self.id_col], sequence)

        if self.include_id_col:
            return record[self.id_col], formatted_sequence, torch.tensor(record[self.label_col], device=nn_utils.get_device())
        else:
            return formatted_sequence, torch.tensor(record[self.label_col], device=nn_utils.get_device())


class ProteinSequenceESM3Dataset(ProteinSequenceDatasetWithID):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col, include_id_col):
        super(ProteinSequenceESM3Dataset, self).__init__(df, id_col, sequence_col, max_seq_len, truncate, label_col)
        self.include_id_col = include_id_col

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]

        if self.include_id_col:
            return record[self.id_col], record[self.sequence_col], torch.tensor(record[self.label_col], device=nn_utils.get_device())
        else:
            return record[self.sequence_col], torch.tensor(record[self.label_col], device=nn_utils.get_device())
