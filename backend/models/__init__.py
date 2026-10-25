"""
models package — SSL model implementations and registry for SSL-MedBench.
"""

from typing import Dict, Type

from .base import BaseSSLModel
from .team1_self_training import SelfTrainingModel
from .team2_label_propagation import LabelPropagationModel
from .team3_co_training import CoTrainingModel
from .team4_s3vm import S3VMModel
from .team5_em_gmm import EMGMMModel

MODEL_REGISTRY: Dict[str, Type[BaseSSLModel]] = {
    "Self-Training (RF)": SelfTrainingModel,
    "Label Propagation": LabelPropagationModel,
    "Co-Training (SVM)": CoTrainingModel,
    "S3VM (TSVM)": S3VMModel,
    "EM + GMM": EMGMMModel,
}

__all__ = [
    "BaseSSLModel",
    "SelfTrainingModel",
    "LabelPropagationModel",
    "CoTrainingModel",
    "S3VMModel",
    "EMGMMModel",
    "MODEL_REGISTRY",
]
