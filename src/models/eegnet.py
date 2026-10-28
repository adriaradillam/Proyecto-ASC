"""
EEGNet: Compact Convolutional Neural Network for EEG-based BCI.
Applications of Soft Computing (ASC), Checkpoint 2 — Representation 2.

This implementation faithfully reproduces EEGNet as described in:
    Lawhern et al. (2018). EEGNet: a compact convolutional neural network
    for EEG-based brain–computer interfaces. Journal of Neural Engineering,
    15(5), 056013. DOI: 10.1088/1741-2552/aace8c

Architecture (adapted for seizure prediction):
  Input: (batch, 1, n_channels, n_samples) — raw filtered EEG tensor

  Block 1 — Temporal Feature Extraction:
    Conv2D(F1 filters, kernel=(1, 64), padding='same', bias=False)
    BatchNorm2D
    DepthwiseConv2D(depth_multiplier=D, kernel=(n_channels, 1), bias=False)
    BatchNorm2D -> ELU -> AveragePool2D(1, 4) -> Dropout(p)

  Block 2 — Separable Temporal Convolution:
    SeparableConv2D(F2 filters, kernel=(1, 16), padding='same', bias=False)
    BatchNorm2D -> ELU -> AveragePool2D(1, 8) -> Dropout(p)

  Classification Head:
    Flatten -> Linear(n_hidden, n_classes) -> Sigmoid

Key design principles preserved from the paper:
  - Depthwise convolution learns a spatial filter per temporal feature map.
  - Separable convolution decouples temporal and spatial processing.
  - MaxNorm weight constraints prevent gradient explosion on small datasets.
  - Dropout applied after each pooling to regularise the compact feature space.

For seizure prediction, the input window is 10 seconds at 128 Hz
(n_samples = 1280), which is fed as the raw EEG tensor after bandpass
filtering in the preprocessing pipeline.

References:
    Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural
    network for EEG-based brain–computer interfaces. J Neural Eng, 15(5).
    DOI: 10.1088/1741-2552/aace8c

    Jiang, Y., et al. (2023). Front Neurosci, 17, 1191683.
    DOI: 10.3389/fnins.2023.1191683
"""

import math
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# MaxNorm Weight Constraint Helper
# ---------------------------------------------------------------------------

class MaxNormConstraint:
    """
    Applies max-norm weight constraint after each gradient step.
    Clips the L2 norm of weight rows to a maximum value `max_norm`.
    As described in Lawhern et al. (2018), this stabilises training of
    the depthwise and separable convolution layers.
    """

    def __init__(self, max_norm: float = 1.0, dim: int = 0):
        self.max_norm = max_norm
        self.dim = dim

    def __call__(self, module: nn.Module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            norms = w.norm(2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            desired = torch.clamp(norms, max=self.max_norm)
            w.mul_(desired / norms)


# ---------------------------------------------------------------------------
# EEGNet Architecture
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """
    Compact EEGNet for binary EEG classification (seizure prediction).

    Args:
        n_channels: Number of EEG channels (default: 18 for CHB-MIT bipolar montage).
        n_samples: Number of time samples per window (default: 1280 = 10s × 128Hz).
        F1: Number of temporal filters in Block 1 (default: 8).
        D: Depth multiplier for depthwise convolution (default: 2 → F2 = F1*D = 16).
        dropout_rate: Dropout probability after each pooling layer (default: 0.5).
        n_classes: Number of output classes (default: 2 for binary prediction).

    Input shape: (batch_size, 1, n_channels, n_samples)
    Output shape: (batch_size, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 18,
        n_samples: int = 1280,     # 10 s × 128 Hz
        F1: int = 8,
        D: int = 2,
        dropout_rate: float = 0.5,
        n_classes: int = 2
    ):
        super(EEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        self.dropout_rate = dropout_rate

        # ------------------------------------------------------------------
        # Block 1: Temporal + Depthwise Spatial Convolution
        # ------------------------------------------------------------------

        # Temporal convolution: learns F1 spectral filters across time.
        # Kernel size = sampling_rate / 2 = 64 (for fs=128 Hz), spans 0.5 s.
        self.block1_temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 64),     # (spatial, temporal)
                padding=(0, 32),         # 'same' padding along temporal axis
                bias=False
            ),
            nn.BatchNorm2d(F1)
        )

        # Depthwise convolution: learns D spatial filters per temporal filter.
        # Acts as a learned spatial (electrode) filter — equivalent to common
        # spatial patterns (CSP) but learned end-to-end.
        self.block1_depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=self.F2,
                kernel_size=(n_channels, 1),  # spans all channels
                groups=F1,                     # one filter group per F1 map
                bias=False
            ),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )

        # ------------------------------------------------------------------
        # Block 2: Separable Temporal Convolution
        # ------------------------------------------------------------------
        # Separable = depthwise (channel-wise) + pointwise (mixing).
        # Reduces parameter count relative to a standard Conv2D.
        self.block2 = nn.Sequential(
            # Depthwise part: temporal filter per feature map
            nn.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, 16),
                padding=(0, 8),         # 'same' along temporal axis
                groups=self.F2,
                bias=False
            ),
            # Pointwise part: mix feature maps
            nn.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        # ------------------------------------------------------------------
        # Classification Head
        # ------------------------------------------------------------------
        # Compute the flattened feature dimension dynamically
        self._flatten_dim = self._compute_flatten_dim()

        self.classifier = nn.Linear(self._flatten_dim, n_classes)

        # Apply MaxNorm constraint to depthwise and classifier weights
        self._max_norm_constraint_dw = MaxNormConstraint(max_norm=1.0, dim=0)
        self._max_norm_constraint_cls = MaxNormConstraint(max_norm=0.25, dim=0)

    def _compute_flatten_dim(self) -> int:
        """Infer the flattened size after the two convolutional blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.block1_temporal(dummy)
            x = self.block1_depthwise(x)
            x = self.block2(x)
            return int(x.numel())

    def apply_max_norm(self):
        """Call after each backward pass to enforce MaxNorm on sensitive weights."""
        dw_conv = self.block1_depthwise[0]
        self._max_norm_constraint_dw(dw_conv)
        self._max_norm_constraint_cls(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG tensor of shape (batch, 1, n_channels, n_samples).

        Returns:
            logits: (batch, n_classes)
        """
        x = self.block1_temporal(x)
        x = self.block1_depthwise(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)       # Flatten
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Training & Inference Wrapper
# ---------------------------------------------------------------------------

class EEGNetTrainer:
    """
    Training and inference wrapper for EEGNet, compatible with the LOSO harness.

    Trains with class-balanced cross-entropy loss (via pos_weight), Adam
    optimiser, cosine annealing LR schedule, and early stopping on validation
    AUC. Records loss/metric curves for the required learning-curve figures.

    Args:
        n_channels: Number of EEG channels.
        n_samples: Number of time samples per window.
        n_epochs: Maximum training epochs (early stopping may terminate earlier).
        lr: Initial Adam learning rate.
        batch_size: Mini-batch size.
        dropout_rate: EEGNet dropout rate.
        patience: Early stopping patience (in epochs, measured on val AUC).
        device: 'cuda', 'cpu', or None for auto-detection.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_channels: int = 18,
        n_samples: int = 1280,
        n_epochs: int = 150,
        lr: float = 1e-3,
        batch_size: int = 32,
        dropout_rate: float = 0.5,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42
    ):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[EEGNet] = None
        self.train_losses: list = []
        self.val_aucs: list = []
        self._best_val_auc: float = -1.0
        self._patience_counter: int = 0
        self._best_state: Optional[dict] = None

    def _build_model(self) -> EEGNet:
        return EEGNet(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            dropout_rate=self.dropout_rate
        ).to(self.device)

    def _prepare_tensors(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert numpy arrays to EEGNet-compatible tensors.
        X shape: (N, n_channels, n_samples) -> (N, 1, n_channels, n_samples)
        """
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)   # Add channel dim
        y_t = torch.tensor(y, dtype=torch.long) if y is not None else None
        return X_t, y_t

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'EEGNetTrainer':
        """
        Train EEGNet with Adam + cosine annealing + early stopping.

        If X_val / y_val are not provided, 15% of training data is held out.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        # Split off validation set if not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.15,
                stratify=y_train,
                random_state=self.random_state
            )

        X_tr, y_tr = self._prepare_tensors(X_train, y_train)
        X_v,  y_v  = self._prepare_tensors(X_val, y_val)

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        self.model = self._build_model()

        # Compute pos_weight to handle class imbalance
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)],
            dtype=torch.float32,
            device=self.device
        )

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1.0, float(pos_weight.item())],
                device=self.device
            )
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        self.train_losses = []
        self.val_aucs = []
        self._best_val_auc = -1.0
        self._patience_counter = 0
        self._best_state = None

        for epoch in range(self.n_epochs):
            # --- Training phase ---
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                self.model.apply_max_norm()

                epoch_loss += loss.item() * len(y_batch)

            avg_loss = epoch_loss / len(y_train)
            self.train_losses.append(avg_loss)
            scheduler.step()

            # --- Validation phase (AUC) ---
            val_probs = self._predict_proba_tensor(X_v)
            try:
                val_auc = roc_auc_score(y_val, val_probs[:, 1])
            except ValueError:
                val_auc = 0.5
            self.val_aucs.append(val_auc)

            # --- Early stopping ---
            if val_auc > self._best_val_auc + 1e-4:
                self._best_val_auc = val_auc
                self._patience_counter = 0
                self._best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    break

        # Restore best weights
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return self

    def _predict_proba_tensor(self, X_t: torch.Tensor) -> np.ndarray:
        """Run inference on a GPU/CPU tensor, return numpy probability array."""
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                batch = X_t[start:start + self.batch_size].to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates (N, 2) for numpy array input."""
        if self.model is None:
            raise RuntimeError("EEGNetTrainer: call fit() before predict_proba().")
        X_t, _ = self._prepare_tensors(X)
        return self._predict_proba_tensor(X_t)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class label with an optional decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_learning_curves(self) -> dict:
        """Return training loss and validation AUC curves (list of epoch values)."""
        return {
            'train_loss': self.train_losses,
            'val_auc': self.val_aucs
        }

    def get_params(self) -> dict:
        return {
            'model': 'EEGNet',
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'n_epochs': self.n_epochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
            'patience': self.patience,
            'device': str(self.device)
        }
