from utils import nn_utils
from models.protein_sequence_classification import ProteinSequenceClassification
from transformers import T5Tokenizer, T5EncoderModel
import torch


class ProtT5_VirusHostPrediction(ProteinSequenceClassification):
    """
    Fine tuning ProtT5 (https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) for Virus Host Prediction
    """

    def __init__(self, input_dim, hidden_dim, n_mlp_layers, n_classes, pre_trained_model_link, hugging_face_cache_dir):
        super(ProtT5_VirusHostPrediction, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes,
                                                         batch_norm=True)
        self.tokenizer, self.pre_trained_model = self.initialize_pre_trained_model(pre_trained_model_link,
                                                                                   hugging_face_cache_dir)
        self.pre_trained_model.eval()

    def initialize_pre_trained_model(self, pre_trained_model_link, hugging_face_cache_dir):
        tokenizer = T5Tokenizer.from_pretrained(pre_trained_model_link, do_lower_case=False,
                                                cache_dir=hugging_face_cache_dir)

        # explicitly set the model to float16 otherwise the generated embeddings will be full precision
        pre_trained_model = T5EncoderModel.from_pretrained(pre_trained_model_link,
                                                           cache_dir=hugging_face_cache_dir)
        return tokenizer, pre_trained_model.to(nn_utils.get_device())

    def get_embedding(self, X):
        sequences, sequence_lengths = X
        token_encoding = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(token_encoding["input_ids"]).to(nn_utils.get_device())
        attention_mask = torch.tensor(token_encoding["attention_mask"]).to(nn_utils.get_device())

        embedding_representation = self.pre_trained_model(input_ids, attention_mask=attention_mask)

        # get per-sequence embedding by averaging the embedding of only the amino acid tokens (excluding padding and any special tokens)
        sequence_embeddings = []
        for i, seq_length in enumerate(sequence_lengths):
            sequence_embedding = embedding_representation.last_hidden_state[i, 0: seq_length + 1].mean(0)
            sequence_embeddings.append(sequence_embedding)

        return torch.stack(sequence_embeddings)

    def get_model(model_params) -> ProteinSequenceClassification:
        # explicitly set the default dtype to float16
        # torch.set_default_dtype(torch.float16)

        model = ProtT5_VirusHostPrediction(input_dim=model_params["input_dim"],
                                           hidden_dim=model_params["hidden_dim"],
                                           n_mlp_layers=model_params["n_mlp_layers"],
                                           n_classes=model_params["n_classes"],
                                           pre_trained_model_link=model_params["pre_trained_model_link"],
                                           hugging_face_cache_dir=model_params["hugging_face_cache_dir"])
        print(model)
        print("BERT_VirusHostPrediction: Number of parameters = ",
              sum(p.numel() for p in model.parameters() if p.requires_grad))

        if nn_utils.get_device() == "cpu":
            print("Casting model to full precision for running on CPU.")
            model.to(torch.float32)
        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])
