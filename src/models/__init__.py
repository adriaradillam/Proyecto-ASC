"""Models package."""
from .baseline import LogisticRegressionBaseline, RandomChancePredictor
from .classical import build_random_forest, build_svm, build_mlp
from .deep_eegnet import EEGNet

__all__ = [
    "LogisticRegressionBaseline",
    "RandomChancePredictor",
    "build_random_forest",
    "build_svm",
    "build_mlp",
    "EEGNet"
]
