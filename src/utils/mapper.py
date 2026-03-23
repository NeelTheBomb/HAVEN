from models.baseline.nlp.fnn import FNN_VirusHostPrediction
from models.baseline.nlp.cnn1d import CNN_1D_VirusHostPrediction
from models.baseline.nlp.rnn import RNN_VirusHostPrediction
from models.baseline.nlp.lstm import LSTM_VirusHostPrediction
from models.baseline.nlp.transformer_encoder import TransformerEncoderVirusHostPrediction

from models.haven.haven import HAVEN
from models.haven.ablation.haven_wo_hierattn import HAVEN_wo_HierAttn

from models.haven.ablation.bert_virus_host_prediction import BERT_VirusHostPrediction
from models.external.prostt5_host_prediction import ProstT5_VirusHostPrediction
from models.external.prott5_host_prediction import ProtT5_VirusHostPrediction
from models.external.esm2_host_prediction import ESM2_VirusHostPrediction
#from models.external.esm3_host_prediction import ESM3_VirusHostPrediction

from datasets.protein_sequence_custom_dataset import ProteinSequenceProstT5Dataset
from datasets.protein_sequence_custom_dataset import ProteinSequenceProtT5Dataset
from datasets.protein_sequence_custom_dataset import ProteinSequenceESM2Dataset

from datasets.collations.custom_collate_function import ESM2CollateFunction

from pipelines.virus_host_prediction_training import fine_tuning_pipeline, fine_tuning_external_pipeline, baseline_deep_learning_pipeline, baseline_machine_learning_pipeline, baseline_homology_pipeline
from pipelines.transfer_learning import masked_language_modeling_pipleine
from pipelines.analysis import perturbation_analysis_pipeline, perturbation_analysis_external_pipeline, embedding_generation_pipeline, virus_host_prediction_testing_pipeline, virus_host_prediction_testing_external_pipeline
from pipelines.few_shot_learning import few_shot_learning_host_prediction_pipeline, few_shot_learning_host_prediction_external_pipeline
from pipelines.evaluation import evaluation_pipeline

pipeline_mapper = {
    "masked_language_modeling": masked_language_modeling_pipleine,
    "virus_host_prediction": fine_tuning_pipeline,
    "virus_host_prediction_external": fine_tuning_external_pipeline,
    "virus_host_prediction_baseline_deep_learning": baseline_deep_learning_pipeline,
    "virus_host_prediction_baseline_machine_learning": baseline_machine_learning_pipeline,
    "virus_host_prediction_baseline_similarity": baseline_similarity_pipeline,
    "virus_host_prediction_test": virus_host_prediction_testing_pipeline,
    "virus_host_prediction_test_external": virus_host_prediction_testing_external_pipeline,
    "few_shot_learning": few_shot_learning_host_prediction_pipeline,
    "few_shot_learning_external": few_shot_learning_host_prediction_external_pipeline,
    "evaluation": evaluation_pipeline,
    "perturbation": perturbation_analysis_pipeline,
    "perturbation_external": perturbation_analysis_external_pipeline,
    "embedding_generation": embedding_generation_pipeline,
}

# mappings of all classes
model_map = {
    "FNN": FNN_VirusHostPrediction,
    "CNN": CNN_1D_VirusHostPrediction,
    "RNN": RNN_VirusHostPrediction,
    "LSTM": LSTM_VirusHostPrediction,
    "Transformer_Encoder": TransformerEncoderVirusHostPrediction,
    "BERT": BERT_VirusHostPrediction,
    "HAVEN_wo_Hierarchical_Attention": HAVEN_wo_HierAttn,
    "HAVEN": HAVEN,
    "ProstT5": ProstT5_VirusHostPrediction,
    "ProtT5": ProtT5_VirusHostPrediction,  # IMP: note the difference between ProstT5 and ProtT5
    "ESM2": ESM2_VirusHostPrediction,
#    "ESM3": ESM3_VirusHostPrediction
}

dataset_map = {
    "ProstT5": ProteinSequenceProstT5Dataset,
    "ProtT5": ProteinSequenceProtT5Dataset,  # IMP: note the difference between ProstT5 and ProtT5
    "ESM2": ProteinSequenceESM2Dataset,
#    "ESM3": ProteinSequenceESM3Dataset
}

collate_function_map = {
    "ESM2": ESM2CollateFunction
}
