# +
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), "..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..", "..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..", "..", "..", "..", ".."))
sys.path.append(os.path.join(os.getcwd(), "..", ".."))

#sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path
# -

import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import statistics
import math
import numpy as np

from src.models.baseline.nlp.transformer.transformer import TransformerEncoder
from src.models.haven import VirProBERT
from src.utils import constants, nn_utils, utils
from src.datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID
from src.datasets.collations.padding_with_id import PaddingWithID


input_file_path = os.path.join(os.getcwd(), "..","..", "..", "..", "input/data/coronaviridae/20240313/sarscov2/uniprot/variants/sarscov2_variants_s.csv")
input_df = pd.read_csv(input_file_path)
input_df

wiv04_input_df = input_df[input_df["id"] == "WIV04"]
wiv04_input_df

# ### Load the model

# +
# Transformer Encoder

pre_train_encoder_settings = {
    "n_heads": 8,
    "depth": 6,
    "input_dim": 512, # input embedding dimension
    "hidden_dim": 1024,
    "max_seq_len": 256,
    "cls_token": True,
    "vocab_size": constants.VOCAB_SIZE
}

pre_trained_encoder_model = TransformerEncoder.get_transformer_encoder(pre_train_encoder_settings, pre_train_encoder_settings["cls_token"])

# +
# HAVEN model
virprobert_settings = {
    "n_mlp_layers": 2,
    "n_classes": 8,
    "input_dim": 512, # input embedding dimension,
    "hidden_dim": 1024,
    "n_heads": 8,
    "stride": 64,
    "cls_token": True,
    "segment_len": pre_train_encoder_settings["max_seq_len"],
    "data_parallel": False,
    "pre_trained_model": pre_trained_encoder_model
}

virprobert_model = VirProBERT.get_model(virprobert_settings)

model_path = os.path.join(os.getcwd(), "..","..", "..", "..",  "output/raw/coronaviridae_s_prot_uniref90_embl_vertebrates_t0.01_c8/20240828/host_multi/fine_tuning_hybrid_cls/mlm_tfenc_l6_h8_lr1e-4_uniref90viridae_msl256b512_ae_bn_vs30cls_s64_hybrid_attention_s64_fnn_2l_d1024_lr1e-4_itr4.pth")
virprobert_model.load_state_dict(torch.load(model_path, map_location=nn_utils.get_device()))

# +
# Load dataset
sequence_settings = {
    "batch_size": 1,
    "id_col": "id",
    "sequence_col": "seq",
    "truncate": False,
    "split": False,
    "feature_type": "token"
}

label_settings = {
    "label_col": "virus_host_name",
    "exclude_labels": [ "nan"],
    "label_groupings": {
      "Chicken": [ "gallus gallus" ],
      "Human": [ "homo sapiens" ],
      "Cat": [ "felis catus" ],
      "Pig": [ "sus scrofa" ],
      "Gray wolf": [ "canis lupus" ],
      "Horshoe bat": ["rhinolophus sp."],
      "Ferret": ["mustela putorius"],
      "Chinese rufous horseshoe bat": ["rhinolophus sinicus"],
    }
}

# +
wiv04_input_df, index_label_map = utils.transform_labels(wiv04_input_df, label_settings, classification_type="multi")

dataset = ProteinSequenceDatasetWithID(wiv04_input_df, 
                                       id_col=sequence_settings["id_col"], 
                                       sequence_col=sequence_settings["sequence_col"], 
                                       max_seq_len=pre_train_encoder_settings["max_seq_len"], 
                                       truncate=sequence_settings["truncate"], 
                                       label_col=label_settings["label_col"])
    
dataset_loader = DataLoader(dataset=dataset, 
                            batch_size=sequence_settings["batch_size"], 
                            shuffle=True,
                            collate_fn=PaddingWithID(pre_train_encoder_settings["max_seq_len"]))
idx_label_map={0: 'Cat', 1: 'Chicken', 2: 'Chinese rufous horseshoe bat', 3: 'Ferret', 4: 'Gray wolf', 5: 'Horshoe bat', 6: 'Human', 7: 'Pig'}
# -

virprobert_model.eval()
seq_id, seq, label = next(iter(dataset_loader))

print(f"seq_id = {seq_id}")
print(f"seq = {seq}, seq_len = {seq.shape}")
print(f"label = {label}")

output = virprobert_model(seq)
output

output_prob = F.softmax(output, dim=-1)
result_df = pd.DataFrame(output_prob.detach().cpu().numpy())
result_df["id"] = seq_id
result_df["y_true"] = label.detach().cpu().numpy()
result_df.rename(columns=idx_label_map)

# +
seq_len = seq.shape[1]
pos_mapping_range = {}
pos_mapping = {}
j = 0
for i in range(0, seq_len + 1, 64):
    start = i
    # end = seq_len if i+127 > seq_len else i+127
    end = i + 256
    if end >= seq_len:
        break
    pos_mapping[j] = f"{j}: {start + 1}-{end}"
    pos_mapping_range[j] = [start, end]
    j += 1
    
    
pos_mapping


# +
def compute_virprobert_embedding(X, WIV04_idx):
    batch_size = X.shape[0]
    X = X.unfold(dimension=1, size=virprobert_model.segment_len, step=virprobert_model.stride)
    X = X.contiguous().view(-1, virprobert_model.segment_len)
    cls_tokens = torch.full(size=(X.shape[0], 1), fill_value=constants.CLS_TOKEN_VAL,
                                    device=nn_utils.get_device())
    
    X = torch.cat([cls_tokens, X], dim=1)
    X = virprobert_model.pre_trained_model(X, mask=None)

    intra_seg_attn_last_enc_layer = virprobert_model.pre_trained_model.encoder.layers[-1].self_attn.self_attn
    
    X = X.view(batch_size, -1, virprobert_model.segment_len + 1, virprobert_model.input_dim)
    X = X[:, :, 0, :]
    X = virprobert_model.self_attn(X, X, X)
    inter_seg_attn_regen = virprobert_model.self_attn.self_attn[WIV04_idx, :, :, :]
    
    return inter_seg_attn_regen, intra_seg_attn_last_enc_layer
    
def compute_cumulative_attention(seq, virprobert_model, WIV04_idx):
    inter_seg_attn = virprobert_model.self_attn.self_attn[WIV04_idx, :, :, :]
    print(f"inter_seg_attn shape = {inter_seg_attn.shape}")
    
    mean_inter_seg_attn = inter_seg_attn.mean(dim=0)
    print(f"mean_inter_seg_attn shape = {mean_inter_seg_attn.shape}")
    n_segments = mean_inter_seg_attn.shape[0]
    print(f"n_segments = {n_segments}")
    
    
    seq_len = seq.shape[-1]
    print(f"seq_len = {seq_len}")
    X = seq
    
    inter_seg_attn_regen, intra_seg_attn_last_enc_layer = compute_virprobert_embedding(X, WIV04_idx)
    print(f"inter_seg attention value equality check = {torch.equal(inter_seg_attn, inter_seg_attn_regen)}")
    
    return inter_seg_attn_regen, mean_inter_seg_attn, intra_seg_attn_last_enc_layer
# -

WIV04_idx = 0
inter_seg_attn_regen, mean_inter_seg_attn, intra_seg_attn_last_enc_layer = compute_cumulative_attention(seq, virprobert_model, WIV04_idx)

# +
plt.clf()
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40
plt.rcParams.update({'font.size': 40})
c

c = 0
for i in range(4):
    for j in range(2):
        df = pd.DataFrame(inter_seg_attn_regen[c].squeeze().detach().cpu().numpy())
        df.rename(columns=pos_mapping, inplace=True)
        df.rename(index=pos_mapping, inplace=True)
        sns.heatmap(df, ax=axs[i, j], linewidth=.1, cmap="crest")
        axs[i, j].set_title(f"Head {c}")
        c += 1

plt.tight_layout(pad=.1)
plt.show()

# +
plt.clf()
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(1, 1, figsize=(8, 8), sharex=False, sharey=True)

df = pd.DataFrame(mean_inter_seg_attn.detach().cpu().numpy())
df.rename(columns=pos_mapping, inplace=True)
df.rename(index=pos_mapping, inplace=True)
sns.heatmap(df, ax=axs, linewidth=.1, cmap="crest")
axs.set_title(f"Mean across all Heads")

plt.show()
# -

intra_seg_attn_last_enc_layer.shape

# mean across all attention heads
intra_seg_attn_last_enc_layer_mean_attn = intra_seg_attn_last_enc_layer.mean(dim=1)
intra_seg_attn_last_enc_layer_mean_attn.shape

# attention values of the class token for all other tokens
intra_seg_attn_last_enc_layer_mean_attn_cls_token = intra_seg_attn_last_enc_layer_mean_attn[:, 0, 1:]
intra_seg_attn_last_enc_layer_mean_attn_cls_token.shape

inter_seg_attn_mean = mean_inter_seg_attn.mean(dim=0)
inter_seg_attn_mean

pos_mapping_range

pos_attn_vals = {}
pos_attn_mean_vals = []
for i in range(seq_len):
    # print(f"i = {i}")
    # traverse the pos_segment map
    for segment_idx, segment in pos_mapping_range.items():
        if i >= segment[0] and i < segment[1]:
            # print(f"i={i} in segment = {segment_idx}:{segment}")
            inter_seg_attn_mean_val = inter_seg_attn_mean[segment_idx]
            # print(f"inter_seg_attn_mean_val = {inter_seg_attn_mean_val}")
            
            intra_seg_idx = range(segment[0], segment[1]).index(i)
            # print(f"intra_seg_idx = {intra_seg_idx}")
            intra_seg_attn_mean_val = intra_seg_attn_last_enc_layer_mean_attn_cls_token[segment_idx][intra_seg_idx]
            
            # print(f"intra_seg_attn_mean_val = {intra_seg_attn_mean_val}")
            pos_attn_val = inter_seg_attn_mean_val.item() * intra_seg_attn_mean_val.item()
            
            if i in pos_attn_vals:
                pos_attn_vals[i].append(pos_attn_val)
            else:
                pos_attn_vals[i] = [pos_attn_val]
for _, attn_vals in pos_attn_vals.items():
    pos_attn_mean_vals.append(statistics.mean(attn_vals))
print(f"pos_attn_mean_vals len = {len(pos_attn_mean_vals)}")

# +
plt.clf()
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(1, 1, figsize=(40, 5), sharex=False, sharey=False)

sns.scatterplot(pos_attn_mean_vals, ax=axs, s=100)
axs.set_xticks(range(0, len(pos_attn_mean_vals), 10)) 
plt.xticks(rotation=90)
plt.show()
# -

# write the mean attention values
output_file_path = os.path.join(os.getcwd(), "..","..", "..", "..", "output/raw/coronaviridae_s_prot_uniref90_embl_vertebrates_t0.01_c8/20240828/host_multi/fine_tuning_hybrid_cls/mlm_tfenc_l6_h8_lr1e-4_uniref90viridae_msl256b512_ae_bn_vs30cls_s64_hybrid_attention_s64_fnn_2l_d1024_lr1e-4_wiv04_mean_attn_vals_last_layer.txt")
np.savetxt(output_file_path, np.array(pos_attn_mean_vals))

# ! pip install logomaker

import logomaker as lm

seq_aligned_df = pd.read_csv(os.path.join(os.getcwd(), "..","..", "..", "..", "input/data/coronaviridae/20240313/wiv04/sarscov2_variants_s_aligned.csv"))
sequences = list(seq_aligned_df["seq"].values)
sequences

seqs_counts_df = lm.alignment_to_matrix(sequences=sequences, to_type="probability", characters_to_ignore=".-X")
seqs_counts_df.shape

step = 50
for i in range(0, seqs_counts_df.shape[0], step):
    start = i
    end = i + step
    positions = list(range(start, end))
    x = seqs_counts_df.reset_index()
    x = x[x["pos"].isin(positions)].set_index("pos")
    lm.Logo(x, stack_order='small_on_top')


