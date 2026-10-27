"""
Baseline Models for EEG Seizure Prediction.
Applications of Soft Computing (ASC).

Contains:
1. Logistic Regression baseline on frequency band powers.
2. Poisson Random Chance Predictor (analytical & Monte Carlo).

References:
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
- Schelter et al. (2006), Testing statistical significance of seizure prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, Optional


class LogisticRegressionBaseline:
    """
    Standard L2-penalized Logistic Regression on relative band power features.
    Provides the required non-deep baseline for paired comparisons.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=self.C,
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs',
                random_state=self.random_state
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model on training feature matrix and labels."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (N, 2)."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class with optional custom probability threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class RandomChancePredictor:
    """
    Analytical and empirical Random Predictor benchmark.
    Evaluates whether a model performs statistically better than a random Poisson process.
    
    Formula (Schelter et al. 2006; Kuhlmann et al. 2018):
        P_chance(alarm in SOP) = 1 - exp(-FPR_h * (SOP_min / 60))
    """

    def __init__(self, sop_min: float = 30.0):
        self.sop_min = sop_min
        self.sop_hours = sop_min / 60.0

    def analytical_sensitivity(self, fpr_per_hour: float) -> float:
        """
        Calculates theoretical probability that an alarm occurs within the SOP
        by pure chance given an average false alarm rate per hour (lambda).
        """
        prob = 1.0 - np.exp(-fpr_per_hour * self.sop_hours)
        return float(np.clip(prob, 0.0, 1.0))

    def evaluate_significance(
        self,
        observed_sensitivity: float,
        observed_fpr_per_hour: float,
        n_seizures: int,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Tests if observed sensitivity exceeds chance level at significance level alpha (binomial test).
        """
        from scipy.stats import binomtest

        p_chance = self.analytical_sensitivity(observed_fpr_per_hour)
        k_success = int(round(observed_sensitivity * n_seizures))

        test_res = binomtest(k_success, n_seizures, p_chance, alternative='greater')
        
        return {
            "p_chance": p_chance,
            "observed_sensitivity": observed_sensitivity,
            "p_value": test_res.pvalue,
            "is_significant": test_res.pvalue < alpha,
            "alpha": alpha
        }
