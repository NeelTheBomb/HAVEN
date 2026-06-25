from utils import nn_utils
from models.protein_sequence_classification import ProteinSequenceClassification
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.utils.rnn import pad_sequence


class LucaVirus_VirusHostPrediction(ProteinSequenceClassification):
    """
    Fine tuning LucaVirus for Virus Host Prediction
    """
    def __init__(self, input_dim, hidden_dim, n_mlp_layers, n_classes, pre_trained_model_link,
                 hugging_face_cache_dir):
        super(LucaVirus_VirusHostPrediction, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes,
                                                            batch_norm=True)
        self.tokenizer, self.pre_trained_model = self.initialize_pre_trained_model(pre_trained_model_link,
                                                                                   hugging_face_cache_dir)
        self.pre_trained_model.eval()

    def initialize_pre_trained_model(self, pre_trained_model_link, hugging_face_cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_link,
                                                 trust_remote_code=True,
                                                 do_lower_case=False,
                                                 cache_dir=hugging_face_cache_dir)
        pre_trained_model = AutoModel.from_pretrained(pre_trained_model_link,
                                                      task_level="token_level",
                                                      task_type="embedding",
                                                      trust_remote_code=True,
                                                      cache_dir=hugging_face_cache_dir)
        return tokenizer, pre_trained_model.to(nn_utils.get_device())

    def get_embedding(self, X):
        # X can be one of several batch formats produced by DataLoader/collate:
        # 1) list of (id, sequence) tuples: [(id1, seq1), (id2, seq2), ...]
        # 2) list containing two tuples: [(id1, id2, ...), (seq1, seq2, ...)]
        # 3) plain list of sequence strings: [seq1, seq2, ...]
        if len(X) == 0:
            return torch.empty(0, self.linear_ip.in_features, device=nn_utils.get_device())

        # if isinstance(X, (tuple, list)) and len(X) == 2 and isinstance(X[0], (tuple, list)) and isinstance(X[1], (tuple, list)) and all(isinstance(item, str) for item in X[1]):
        #     # X is a batch like (ids, sequences), where the second element is the sequence tuple/list
        #     sequences = list(X[1])
        # elif isinstance(X[0], (tuple, list)) and len(X[0]) == 2 and isinstance(X[0][1], str):
        #     # X is a list of (id, sequence) pairs
        #     sequences = [sequence for _, sequence in X]
        # else:
        #     # X is already a list of raw sequences
        #     sequences = list(X)

        sequences = list(X[1])
        token_ids_list = []
        attention_masks = []
        for seq in sequences:
            tokenizer_kwargs = {
                "add_special_tokens": True,
                "return_tensors": "pt",
            }
            token_encoding = self.tokenizer(seq, seq_type="prot", **tokenizer_kwargs)

            token_ids_list.append(token_encoding["input_ids"].squeeze(0))
            attention_masks.append(token_encoding["attention_mask"].squeeze(0))

        padding_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=padding_value).to(nn_utils.get_device())
        attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0).to(nn_utils.get_device())

        embedding_representation = self.pre_trained_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = embedding_representation.last_hidden_state

        # Exclude special tokens and padding when averaging.
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if hasattr(self.tokenizer, "all_special_ids"):
            for special_id in self.tokenizer.all_special_ids:
                special_mask |= input_ids == special_id

        valid_mask = attention_mask.bool() & ~special_mask
        valid_mask = valid_mask.float().unsqueeze(-1)

        token_sum = (last_hidden_state * valid_mask).sum(dim=1)
        token_count = valid_mask.sum(dim=1).clamp(min=1)
        sequence_embeddings = token_sum / token_count
        return sequence_embeddings

        # sequence_embeddings = []
        # for seq in sequences:
        #     prot_inputs = self.tokenizer(
        #                     seq,
        #                     # note: protein sequence
        #                     seq_type="prot",
        #                     return_tensors="pt",
        #                     add_special_tokens=True)

        #     new_prot_inputs = {}
        #     for item in prot_inputs.items():
        #         new_prot_inputs[item[0]] = item[1].to(nn_utils.get_device())
        #     prot_inputs = new_prot_inputs

        #     with torch.no_grad():
        #         prot_outputs = self.pre_trained_model(**prot_inputs)
        #         # last hidden matrix as embedding matrix: [batch_size, seq_len + 2, hidden_size]
        #         prot_last_hidden = prot_outputs.last_hidden_state
        #         # mean pooling
        #         mean_prot_embedding = prot_last_hidden[:, 1:-1, :].mean(dim=1)

        #         sequence_embeddings.append(mean_prot_embedding.squeeze(0))

        # return torch.stack(sequence_embeddings)

    def get_model(model_params) -> ProteinSequenceClassification:
        model = LucaVirus_VirusHostPrediction(input_dim=model_params["input_dim"],
                                              hidden_dim=model_params["hidden_dim"],
                                              n_mlp_layers=model_params["n_mlp_layers"],
                                              n_classes=model_params["n_classes"],
                                              pre_trained_model_link=model_params["pre_trained_model_link"],
                                              hugging_face_cache_dir=model_params["hugging_face_cache_dir"])
        print(model)
        print("LucaVirus_VirusHostPrediction: Number of parameters = ",
              sum(p.numel() for p in model.parameters() if p.requires_grad))

        if nn_utils.get_device() == "cpu":
            print("Casting model to full precision for running on CPU.")
            model.to(torch.float32)
        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])
