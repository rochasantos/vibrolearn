import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class CNNLSTMNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # CNN block
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        
        # LSTM block
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=100,
            num_layers=3,
            batch_first=True
        )
        self.fc1 = nn.Sequential(
            nn.Linear(100, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Output
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)

        return x
    

class CNNLSTMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        device="cuda",
        verbose=True
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.device_ = self.device

        self.classes_ = np.unique(y)
        self.class_to_index_ = {
            label: idx for idx, label in enumerate(self.classes_)
        }
        self.index_to_class_ = {
            idx: label for label, idx in self.class_to_index_.items()
        }
        y_encoded = np.array(
            [self.class_to_index_[label] for label in y],
            dtype=np.int64
        )

        self.model_ = CNNLSTMNet(
            num_classes=len(self.classes_)
        ).to(self.device_)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_encoded, dtype=torch.long)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate
        )

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                outputs = self.model_(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch [{epoch + 1}/{self.epochs}] - Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()

        predictions = []

        loader = DataLoader(
            torch.tensor(X, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device_)
                outputs = self.model_(xb)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array([self.index_to_class_[idx] for idx in predictions])

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.asarray(y))