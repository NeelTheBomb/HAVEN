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
        # X is a sequence batch like ('SEQ1', 'SEQ2', ...)
        if len(X) == 0:
            return torch.empty(0, self.linear_ip.in_features, device=nn_utils.get_device())

        sequences = list(X)

        token_ids_list = []
        seq_lengths = []
        for seq in sequences:
            token_encoding = self.tokenizer(seq, seq_type="prot", return_tensors="pt", add_special_tokens=True)
            token_ids = token_encoding["input_ids"].squeeze(0)
            token_ids_list.append(token_ids)
            seq_lengths.append(token_ids.size(0))

        padding_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=padding_value).to(nn_utils.get_device())
        attention_mask = (input_ids != padding_value).to(nn_utils.get_device())

        #feeding batch of sequences with attention_mask through pre_trained model faster than feeding each sequence individually
        embedding_representation = self.pre_trained_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = embedding_representation.last_hidden_state

        sequence_embeddings = []
        for batch_index, length in enumerate(seq_lengths):
            #mean pooling over embeddings of amino acid tokens (excluding padding, start, and end tokens)
            sequence_embedding = last_hidden_state[batch_index, 1: length - 1, :].mean(dim=0)
            sequence_embeddings.append(sequence_embedding)

        return torch.stack(sequence_embeddings)

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
