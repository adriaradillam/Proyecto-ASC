"""Models package — Checkpoints 1 & 2."""
from .baseline import LogisticRegressionBaseline, RandomChancePredictor
from .classical import RandomForestSeizurePredictor, SVMSeizurePredictor, MLPSeizurePredictor
from .eegnet import EEGNet, EEGNetTrainer

__all__ = [
    # Checkpoint 1
    "LogisticRegressionBaseline",
    "RandomChancePredictor",
    # Checkpoint 2 — classical
    "RandomForestSeizurePredictor",
    "SVMSeizurePredictor",
    "MLPSeizurePredictor",
    # Checkpoint 2 — deep learning
    "EEGNet",
    "EEGNetTrainer",
]
