import os
import sys
import pathlib
# Ensure repository src directory is on sys.path for local imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from models.external.lucavirus_host_prediction import LucaVirus_VirusHostPrediction

print('IMPORT_ATTEMPT')
model = LucaVirus_VirusHostPrediction(
    input_dim=2560,
    hidden_dim=1024,
    n_mlp_layers=2,
    n_classes=5,
    pre_trained_model_link='LucaGroup/LucaVirus-default-step3.8M',
    hugging_face_cache_dir='output/cache_dir'
)
print('MODEL_OK')
emb = model.get_embedding([('id1','MKTLLILTAVVLL')])
print('EMB_SHAPE', emb.shape)
print('EMB_DTYPE', emb.dtype)
