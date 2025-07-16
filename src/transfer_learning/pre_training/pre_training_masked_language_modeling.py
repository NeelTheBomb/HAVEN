import torch.nn as nn
import torch
from utils import nn_utils, constants


# masked langauge modeling (BERT)
class MaskedLanguageModel(nn.Module):
    def __init__(self, encoder_model, encoder_dim, mask_prob=0.15, random_mask_prob=0.1, no_change_mask_prob=0.1):
        super(MaskedLanguageModel, self).__init__()
        self.encoder_model = encoder_model
        self.pad_token_val = constants.PAD_TOKEN_VAL
        self.mask_token_val = constants.MASK_TOKEN_VAL
        self.no_mask_token_vals = [constants.PAD_TOKEN_VAL, constants.CLS_TOKEN_VAL]
        self.n_tokens = len(constants.AMINO_ACID_VOCABULARY) + 1  # n_tokens = size of amino_acid vocab + 1 (for the pad_token)
        self.mask_prob = mask_prob
        self.random_mask_prob = random_mask_prob
        self.no_change_mask_prob = no_change_mask_prob

        # cross entropy loss expects targets as the class indices which in our case is the same as the amino acid vocab token value
        # pad token val of 0 will be ignored as per the definition of CrossEntropyLoss(ignore_index=pad_token_val) in mlm pipeline.
        self.output_projection = nn.Linear(encoder_dim, self.n_tokens)

    def mask_sequence_batch(self, sequence_batch):
        # create a clone of the original sequence_batch to generate labels for masked positions
        label_batch = sequence_batch.clone()

        # mask <mask_prob> (15%) of the sequence_batch
        init_mask = torch.rand(sequence_batch.shape, device=nn_utils.get_device()) < self.mask_prob

        # exclude the <no_mask_tokens> if selected for masking in mask_pos
        for no_mask_token_val in self.no_mask_token_vals:
            no_mask = sequence_batch != no_mask_token_val # positions WITHOUT the <no_mask_token_val>
            init_mask = init_mask & no_mask # only positions WITHOUT the the <no_mask_token_val> will be retained for final masking

        # TODO: Revisit tho logic of selecting 10% for random replacement and no replacement from sequence_batch and not init_mask
        # mask for positions to be left unchanged (i.e., replace with the original tokens)
        unchanged_token_mask = torch.rand(sequence_batch.shape, device=nn_utils.get_device()) < self.no_change_mask_prob
        unchanged_token_mask = init_mask & unchanged_token_mask

        ## Masking: Replace with Random Tokens
        # mask for positions to be replaced with random tokens
        random_token_mask = torch.rand(sequence_batch.shape, device=nn_utils.get_device()) < self.random_mask_prob
        random_token_mask = init_mask & random_token_mask

        # positions for random masking
        # returns indices of all non-zero values in the tensor
        # as_tuple=True Returns a tuple of 1-D tensors, one for each dimension in input, each containing the indices (in that dimension) of all non-zero elements of input .
        random_mask_pos = torch.nonzero(random_token_mask, as_tuple=True)

        # random tokens to be used for replacement in each of the selected positions
        # low is NOT equal to 0 because 0 is pad token value
        random_mask_tokens = torch.randint(low=1, high=self.n_tokens, size=(len(random_mask_pos[0]), ),
                                           device=nn_utils.get_device(),
                                           dtype=sequence_batch.dtype)
        # replace the random token positions with the generated random tokens
        sequence_batch[random_mask_pos] = random_mask_tokens

        # positions to be masked with mask_token_val
        mask = init_mask & ~random_token_mask & ~unchanged_token_mask

        # fill the mask positions with the mask_token_val
        sequence_batch.masked_fill_(mask, self.mask_token_val)

        ## Replace all the non masked positions (init_mask) in the label with pad_token_val which will be ignored in the Cross Entropy loss calculation  as below
        # this code is in the mlm pipeline: CrossEntropyLoss(ignore_index=pad_token_val)
        label_batch.masked_fill_(~init_mask, self.pad_token_val)

        return sequence_batch, label_batch, init_mask


    def forward(self, X):
        X, label, mask = self.mask_sequence_batch(X)
        seq_emb = self.encoder_model(X, mask)
        masked_seq_logits = self.output_projection(seq_emb)
        return masked_seq_logits, label


def get_mlm_model(encoder_model, mlm_model):
    mlm_model = MaskedLanguageModel(encoder_model=encoder_model,
                                    encoder_dim=mlm_model["encoder_dim"],
                                    mask_prob=mlm_model["mask_prob"],
                                    random_mask_prob=mlm_model["random_mask_prob"],
                                    no_change_mask_prob=mlm_model["no_change_mask_prob"])

    print(mlm_model)
    print("Number of parameters = ", sum(p.numel() for p in mlm_model.parameters() if p.requires_grad))
    # return nn.DataParallel(mlm_model.to(nn_utils.get_device()))
    return mlm_model.to(nn_utils.get_device())
