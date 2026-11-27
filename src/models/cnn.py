"""PyTorch 1D CNN for the project's ordered packet-feature vector."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CNN_ARCHITECTURE_VERSION = "packet-feature-cnn1d-18-32-64-dense-128-64-6-v2"


class PacketCNN(nn.Module):
    """Convolve over the ordered 18 packet fields (not over packet time)."""

    def __init__(self, input_features: int = 18, classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (input_features // 2) + input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, classes),
        )
        self.input_features = int(input_features)
        self.classes = int(classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        conv_features = self.features(x).flatten(1)
        raw_features = x.flatten(1)
        return self.classifier(torch.cat((conv_features, raw_features), dim=1))


class PacketCNNClassifier:
    """Small sklearn-compatible wrapper used by the existing IDS registry."""

    name = "1D CNN"

    def __init__(self, input_features: int = 18, classes: int = 6, epochs: int = 60,
                 batch_size: int = 256, learning_rate: float = 0.0005,
                 patience: int = 8, random_state: int = 42):
        self.name = "1D CNN"
        self.n_features_in_ = int(input_features)
        self.classes_ = np.arange(int(classes), dtype=np.int64)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.model = PacketCNN(self.n_features_in_, len(self.classes_))
        self.training_history_: list[dict[str, float]] = []

    def fit(self, X, y, validation_data=None):
        x = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64)
        self._validate(x, labels)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        train_x = torch.from_numpy(x)
        train_y = torch.from_numpy(labels)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size,
                            shuffle=True, generator=generator)
        counts = np.bincount(labels, minlength=len(self.classes_)).astype(np.float64)
        class_weights = np.sqrt(len(labels) / (len(self.classes_) * np.maximum(counts, 1)))
        class_weights = torch.as_tensor(class_weights / class_weights.mean(), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        val_x = val_y = None
        if validation_data is not None:
            val_x = torch.from_numpy(np.asarray(validation_data[0], dtype=np.float32))
            val_y = torch.from_numpy(np.asarray(validation_data[1], dtype=np.int64))
            self._validate(val_x.numpy(), val_y.numpy())

        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            train_loss = float(np.mean(losses))
            if val_x is None:
                score = train_loss
            else:
                self.model.eval()
                with torch.no_grad():
                    score = float(criterion(self.model(val_x), val_y))
            self.training_history_.append({"epoch": float(epoch + 1), "train_loss": train_loss,
                                           "validation_loss": score})
            if score < best_loss - 1e-5:
                best_loss = score
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if val_x is not None and stale_epochs >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _validate(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected a two-dimensional feature matrix with {self.n_features_in_} columns")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Labels must be a one-dimensional vector matching the feature rows")
        if len(X) == 0 or not np.isfinite(X).all() or not np.isin(y, self.classes_).all():
            raise ValueError("Training data must be finite, non-empty, and use the configured class indices")

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected a two-dimensional feature matrix with {self.n_features_in_} columns")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
