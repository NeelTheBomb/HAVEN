import torch
import torch.nn as nn
import gc
from models.protein_sequence_classification import ProteinSequenceClassification


class PrototypicalNetworkFewShotClassifier(nn.Module):
    def __init__(self, pre_trained_model: ProteinSequenceClassification):
        super(PrototypicalNetworkFewShotClassifier, self).__init__()
        self.pre_trained_model = nn.DataParallel(pre_trained_model)

    def forward(self, support_sequences, support_labels, query_sequences, batch_size):
        # compute prototypes for each label
        prototypes = []
        # unique returns the labels in sorted order
        # we assume the labels are always (0, 1, ...., n_way-1)
        for label in torch.unique(support_labels):
            # assuming n_shot is within the server memory constraints
            # i.e, n_shot <= batch_size
            label_support_features, _ = self.pre_trained_model(
                torch.index_select(support_sequences, dim=0,
                                   index=torch.nonzero(support_labels == label).squeeze()),  # torch.nonzero gives the indices with non-zero elements but it adds a dimension as [n, 1] hence we use squeeze to remove the added extra dimension
                embedding_only=True
            )
            # prototype is the mean of the support features
            prototypes.append(label_support_features.mean(0))
            del label_support_features # mark for deletion
            torch.cuda.empty_cache()

        # memory cleanup
        del support_sequences # mark for deletion
        torch.cuda.empty_cache()
        gc.collect() # garbage collection to free up memory

        # assuming order is maintained and the prototype vector for each label is located at the corresponding index
        prototypes = torch.stack(prototypes) # n_way X embedding_dimension
        # compute output in batches
        self.output = self.compute_output(query_sequences, batch_size, prototypes) # shape n_query X n_way

        # memory cleanup
        del query_sequences # mark for deletion
        torch.cuda.empty_cache()  # empty cachce
        gc.collect() # garbage collection to free up memory

        return self.output

    # method to get compute output for query sequences by generating embeddings for sequences in mini_batches, if required
    def compute_output(self, query_sequences, batch_size, prototypes):
        n_sequences = len(query_sequences)
        output = []
        n_gpus = torch.cuda.device_count()
        for i in range(0, n_sequences, batch_size):
            mini_batch = query_sequences[i: i + batch_size]
            query_features = None

            if mini_batch.shape[0] < n_gpus:
                query_features = self.compute_query_features_with_repetition(mini_batch, n_gpus)
            else:
                query_features, _ = self.pre_trained_model(mini_batch, embedding_only=True)

            output.append(-torch.cdist(query_features, prototypes))

            # cleanup memory and empty cache
            del mini_batch
            del query_features
            torch.cuda.empty_cache()

        return torch.cat(output)

    def compute_query_features_with_repetition(self, mini_batch, n_gpus):
        # if number of samples in the mini_batch is less than the number of gpus available, dataparallel will error out due to insufficient samples to split among the multiple GPUs
        # work around (though hacky):
        # 1. create copies of the mini_batch such that every GPU will have the same mini_batch
        # 2. use only the features from the first GPU for the mini_batch and ignore the embeddings from the remaining GPUs

        # 1 indicates not changing the size in that dimension
        # i.e, number of times for repitition = 1 (technically zero), so no repetition along the columns(second dimension)
        mini_batch = mini_batch.repeat(n_gpus, 1)
        query_features, _ = self.pre_trained_model(mini_batch, embedding_only=True)
        # return only the features for the mini_batch from the first GPU
        # add a batch dimension, i.e. dimension at axis=0 with value=1
        return query_features[0].unsqueeze(0)
