"""
Classical Non-Linear Models for EEG Seizure Prediction (Checkpoint 2).
Applications of Soft Computing (ASC).

Contains:
1. Random Forest Classifier — ensemble of decision trees with balanced class weights.
2. Support Vector Machine (SVM) — RBF kernel, probability calibration via Platt scaling.
3. Multilayer Perceptron (MLP) — shallow ANN with a configurable hidden layer.

All models follow the same sklearn Pipeline API (StandardScaler -> Classifier) so they
plug seamlessly into the Leave-One-Seizure-Out (LOSO) evaluation harness.

References:
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
- Jiang et al. (2023), Front Neurosci (DOI: 10.3389/fnins.2023.1191683)
- Breiman (2001), Random Forests, Mach Learn (DOI: 10.1023/A:1010933404324)
- Chang & Lin (2011), LIBSVM, ACM TIST (DOI: 10.1145/1961189.1961199)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestSeizurePredictor:
    """
    Random Forest ensemble for biomarker-based seizure prediction.

    Uses balanced class weights to compensate for the natural imbalance between
    rare preictal windows and abundant interictal baseline. Feature importances
    are exposed after fitting for interpretability analysis.

    Reference:
        Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
        DOI: 10.1023/A:1010933404324
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight='balanced',
                n_jobs=-1,
                random_state=self.random_state
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestSeizurePredictor':
        """Fit the Random Forest on the training feature matrix."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates (N, 2)."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class label with an optional decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean decrease in impurity (MDI) per feature, from the fitted forest."""
        return self.model.named_steps['clf'].feature_importances_

    def get_params(self) -> Dict[str, Any]:
        return {
            'model': 'RandomForest',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }


# ---------------------------------------------------------------------------
# Support Vector Machine
# ---------------------------------------------------------------------------

class SVMSeizurePredictor:
    """
    Support Vector Machine with an RBF kernel for seizure prediction.

    Probability estimates are obtained via Platt scaling (cross-validated sigmoid
    calibration), which allows ROC-AUC and PR-AUC to be computed on the continuous
    score rather than the hard class boundary.

    The C and gamma hyperparameters are set to sensible defaults; a grid-search
    wrapper is provided in the LOSO harness for optional tuning.

    Reference:
        Chang & Lin (2011). LIBSVM: A library for support vector machines.
        ACM TIST, 2(3), 27. DOI: 10.1145/1961189.1961199
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = 'scale',
        random_state: int = 42
    ):
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                C=self.C,
                kernel='rbf',
                gamma=self.gamma,
                class_weight='balanced',
                probability=True,          # Platt scaling for calibrated probabilities
                random_state=self.random_state
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMSeizurePredictor':
        """Fit the SVM with Platt-calibrated probabilities."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return Platt-scaled class probabilities (N, 2)."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class label with an optional decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        return {
            'model': 'SVM_RBF',
            'C': self.C,
            'gamma': self.gamma,
            'random_state': self.random_state
        }


# ---------------------------------------------------------------------------
# Multilayer Perceptron (shallow ANN)
# ---------------------------------------------------------------------------

class MLPSeizurePredictor:
    """
    Shallow Multilayer Perceptron (ANN) for seizure prediction.

    Architecture: Input -> [256 -> 128] hidden layers with ReLU activation ->
    Output (sigmoid). Trained with Adam and early stopping on a held-out
    validation fraction to prevent overfitting on the small preictal set.

    This model bridges the gap between classical ML (RF, SVM) and deep learning
    (EEGNet) and is required by the course rubric as one of the two model families.

    Reference:
        Jiang et al. (2023). Front Neurosci, 17, 1191683.
        DOI: 10.3389/fnins.2023.1191683
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (256, 128),
        alpha: float = 1e-3,        # L2 regularisation strength
        max_iter: int = 500,
        random_state: int = 42
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=self.alpha,
                batch_size='auto',
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=self.random_state,
                verbose=False
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPSeizurePredictor':
        """Fit the MLP with Adam optimiser and early stopping."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates (N, 2)."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class label with an optional decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_loss_curve(self) -> np.ndarray:
        """Return training loss curve (available after fitting)."""
        return np.array(self.model.named_steps['clf'].loss_curve_)

    def get_validation_scores(self) -> np.ndarray:
        """Return validation score curve (available after fitting with early stopping)."""
        return np.array(self.model.named_steps['clf'].validation_scores_)

    def get_params(self) -> Dict[str, Any]:
        return {
            'model': 'MLP',
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
