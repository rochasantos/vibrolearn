import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class CNN1D(nn.Module):
    """
    Simple 1D CNN for raw vibration segments.
    Input: (B, C, L)
    """
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # -> (B, 128, 1)
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x).squeeze(-1)  # (B, 128)
        return self.classifier(x)


class SklearnCNN1DClassifier(BaseEstimator, ClassifierMixin):
    """
    scikit-learn compatible CNN classifier (fit/predict),
    so you can pass it as `model` into your existing holdout pipeline.

    X expected as:
      - (N, L)  -> will become (N, 1, L)
      - (N, L, C) -> will become (N, C, L)
      - (N, C, L) -> kept as is
    y can be strings (e.g., 'Inner Race'), will be encoded internally.
    """
    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | None = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed
        self.verbose = verbose

        # fitted attrs
        self.label_to_index_ = None
        self.index_to_label_ = None
        self.model_ = None

        self.mean_ = None
        self.std_ = None

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            # (N, L) -> (N, 1, L)
            X = X[:, None, :]
        elif X.ndim == 3:
            # could be (N, L, C) or (N, C, L)
            # Your pipeline usually stores as (N, L, C). We'll detect that.
            N, A, B = X.shape
            # Heuristic: if last dim is small (1..8), assume (N, L, C)
            if B <= 16:
                X = np.transpose(X, (0, 2, 1))  # (N, C, L)
            # else assume already (N, C, L)
        else:
            raise ValueError(f"X must have 2 or 3 dims, got {X.ndim} with shape {X.shape}")
        return X.astype(np.float32)

    def _encode_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if self.label_to_index_ is None:
            labels = np.unique(y)
            self.label_to_index_ = {lab: i for i, lab in enumerate(labels)}
            self.index_to_label_ = {i: lab for lab, i in self.label_to_index_.items()}
        return np.array([self.label_to_index_[lab] for lab in y], dtype=np.int64)

    def fit(self, X, y):
        self._set_seed()
        device = self._get_device()

        Xp = self._prepare_X(X)
        yi = self._encode_y(y)

        N, C, L = Xp.shape
        n_classes = int(np.max(yi) + 1)

        self.model_ = CNN1D(in_channels=C, n_classes=n_classes).to(device)

        ds = TensorDataset(
            torch.from_numpy(Xp),
            torch.from_numpy(yi),
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optim = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        loss_fn = nn.CrossEntropyLoss()

        self.model_.train()

        for ep in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                optim.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

                total_loss += float(loss.item()) * xb.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            avg_loss = total_loss / total
            train_acc = correct / total

            if self.verbose:
                print(
                    f"[CNN] Epoch {ep+1:03d}/{self.epochs} "
                    f"| Loss: {avg_loss:.4f} "
                    f"| Train Acc: {train_acc:.4f}"
                )

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() before predict().")

        device = self._get_device()
        self.model_.eval()

        Xp = self._prepare_X(X)

        ds = TensorDataset(torch.from_numpy(Xp))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device)
                logits = self.model_(xb)
                pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(pred_idx)

        pred_idx = np.concatenate(preds, axis=0)
        # return labels in the same type your metrics expect (strings)
        return np.array([self.index_to_label_[int(i)] for i in pred_idx], dtype=object)