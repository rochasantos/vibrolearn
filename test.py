import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset

from assesment.traintest import performance as holdout
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import numpy as np

from dataset.cwru.sehri_et_al import get_list_of_papers_splits, get_list_of_X_y, load_matlab_acquisition

# CNN Backbone
class CNNBackbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 7)),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(1, 3)),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 1, 1, 1024)
        x = self.features(x)   # -> (B, 256, 1, T)
        return x

# LSTM
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = CNNBackbone()

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 1, 1024)
        x = self.cnn(x)              # (B, 256, 1, T)
        x = x.squeeze(2)             # (B, 256, T)
        x = x.permute(0, 2, 1)       # (B, T, 256)

        lstm_out, _ = self.lstm(x)   # (B, T, 256)

        # Take last timestep
        x = lstm_out[:, -1, :]       # (B, 256)

        x = self.classifier(x)       # (B, num_classes)
        return x


# Train
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(xb)                 # (B, num_classes)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    return (total_loss / total), (correct / total)


def fit(model, train_loader, epochs=5, lr=1e-3, device="cuda"):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | acc={acc:.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    return (total_loss / total), (correct / total)


def fit_with_val(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )


def predict_all(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return y_true, y_pred



def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    segment_length = 1024

    list_of_folds = get_list_of_papers_splits()
    list_of_X_y = get_list_of_X_y(
        list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=['DE'],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition,
        train=True
    )

    Xtr, ytr, Xtr2, ytr2, Xte, yte = list_of_X_y[0][0], list_of_X_y[0][1], list_of_X_y[1][0], list_of_X_y[1][1], list_of_X_y[2][0], list_of_X_y[2][1]
    print(Xtr.shape, ytr.shape, " / ", Xte.shape, yte.shape)

    # --- Label encoding (fit on train labels only) ---
    unique_label = np.unique(ytr)
    label_to_index = {label: idx for idx, label in enumerate(unique_label)}

    # Encode train labels
    ytr_encoded = np.array([label_to_index[label] for label in ytr], dtype=np.int64)

    # Encode test labels using the SAME mapping
    # (Optional safety: ensure test has no unseen labels)
    unseen = set(np.unique(yte)) - set(unique_label)
    if len(unseen) > 0:
        raise ValueError(f"Test set has unseen labels not present in train: {unseen}")

    yte_encoded = np.array([label_to_index[label] for label in yte], dtype=np.int64)

    # --- Convert X to torch and reshape for Conv2d: (N, L, C) -> (N, C, 1, L) ---
    Xtr_torch = torch.from_numpy(Xtr).float().permute(0, 2, 1).unsqueeze(2)
    ytr_torch = torch.from_numpy(ytr_encoded).long()

    Xte_torch = torch.from_numpy(Xte).float().permute(0, 2, 1).unsqueeze(2)
    yte_torch = torch.from_numpy(yte_encoded).long()

    # --- Build datasets ---
    ds_tr = TensorDataset(Xtr_torch, ytr_torch)
    ds_te = TensorDataset(Xte_torch, yte_torch)

    # --- Train/Val split ONLY on train dataset ---
    N = len(ds_tr)
    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(0.8 * N)
    train_idx = idx[:split]
    val_idx   = idx[split:]

    train_ds = Subset(ds_tr, train_idx)
    val_ds   = Subset(ds_tr, val_idx)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=2
    )

    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=2
    )

    test_loader = DataLoader(
        ds_te, batch_size=64, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=2
    )

    # Quick sanity check
    xb, yb = next(iter(test_loader))
    print("Test batch X:", xb.shape, xb.dtype)  # expected: (B, 1, 1, 1024)
    print("Test batch y:", yb.shape, yb.dtype)  # expected: (B,)

    check_cnn = False
    check_cnn_lstm = False
    train_cnn_lstm = False
    validate_cnn_lstm = True
    if check_cnn:
        # Quick sanity check
        xb, yb = next(iter(train_loader))
        print("batch X:", xb.shape)  # expected: (64, 1, 1, 1024)
        print("batch y:", yb.shape)  # expected: (64,)

        cnn = CNNBackbone()
        xb, yb = next(iter(train_loader))
        out = cnn(xb)
        print("CNN out:", out.shape)  # expect: (B, 256, 1, T)

        # Prepare for output CNN to LSTM
        out = out.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        print("CNN out reshaped for LSTM:", out.shape)

    if check_cnn_lstm:
        model = CNNLSTM(num_classes=len(np.unique(ytr_encoded)))
        xb, yb = next(iter(train_loader))

        out = model(xb)
        print("Model output:", out.shape)

    if train_cnn_lstm:
        num_classes = len(np.unique(ytr_encoded))        
        xb, yb = next(iter(train_loader))
        print("xb:", xb.shape, xb.dtype)
        print("yb:", yb.shape, yb.dtype, "min/max:", yb.min().item(), yb.max().item())
        print("num_classes:", num_classes)

        model = CNNLSTM(num_classes=num_classes)

        fit(model, train_loader, epochs=1, lr=1e-3, device=device)

    if validate_cnn_lstm:
        num_classes = len(np.unique(ytr_encoded))
        model = CNNLSTM(num_classes=num_classes)
        fit_with_val(model, train_loader, val_loader, epochs=30, lr=1e-3, device=device)

        y_true, y_pred = predict_all(model, val_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
        print("\nClassification report:\n")
        index_to_label = {v: k for k, v in label_to_index.items()}
        target_names = [index_to_label[i] for i in range(len(index_to_label))]
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

        # Metrics on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"TEST | loss={test_loss:.4f} | acc={test_acc:.4f}")

        y_true_te, y_pred_te = predict_all(model, test_loader, device)
        cm_te = confusion_matrix(y_true_te, y_pred_te)
        print("Test confusion matrix:\n", cm_te)
        index_to_label = {v: k for k, v in label_to_index.items()}
        target_names = [index_to_label[i] for i in range(len(index_to_label))]
        print("\nTest classification report:\n")
        print(classification_report(y_true_te, y_pred_te, target_names=target_names, digits=4))

        # list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]
        # scores = holdout(model, list_of_X_y[0][0], list_of_X_y[0][1], list_of_X_y[1][0], list_of_X_y[1][1], list_of_metrics=list_of_metrics)
        
