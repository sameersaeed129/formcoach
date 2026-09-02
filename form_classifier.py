"""
form_classifier.py
----------------------
An LSTM that classifies a single rep's joint-angle time series into a
form-quality category (good form, knee valgus, insufficient depth, back
rounding).
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from synthetic_data import CLASSES, SEQ_LEN


class FormLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 2,
                 num_classes: int = len(CLASSES), dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, (h_n, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(last_hidden)


class FeatureNormalizer:
    """Per-channel mean/std normalization, fit on training data and reused
    identically at inference time."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        # X: (N, T, C)
        self.mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self.std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.std = np.array(state["std"], dtype=np.float32)


def resample_sequence(seq: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a (T, C) sequence to (target_len, C), so reps of
    different lengths (people move at different speeds) all feed the LSTM
    as fixed-size inputs."""
    t_orig = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        resampled[:, c] = np.interp(t_new, t_orig, seq[:, c])
    return resampled


class FormClassifier:
    """Inference wrapper: loads trained weights + normalizer, runs predictions."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = FormLSTM()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.normalizer = FeatureNormalizer()
        self.normalizer.load_state_dict(checkpoint["normalizer_state"])

    def predict(self, feature_sequence: np.ndarray) -> Tuple[str, np.ndarray]:
        """feature_sequence: (T, 4) array of [knee_angle, hip_angle, torso_lean, knee_valgus].
        Returns (predicted_label, class_probabilities)."""
        resampled = resample_sequence(feature_sequence, SEQ_LEN)
        normalized = self.normalizer.transform(resampled)
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probs))
        return CLASSES[predicted_idx], probs
