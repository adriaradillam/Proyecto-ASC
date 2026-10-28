"""Evaluation package — metrics, LOSO harness, bootstrap comparison."""
from .metrics import (
    compute_standard_metrics,
    compute_seizure_level_metrics,
    paired_bootstrap_test
)
from .loso import run_loso_cv, compare_models_bootstrap

__all__ = [
    "compute_standard_metrics",
    "compute_seizure_level_metrics",
    "paired_bootstrap_test",
    "run_loso_cv",
    "compare_models_bootstrap"
]
