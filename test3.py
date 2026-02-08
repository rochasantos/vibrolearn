import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import multiprocessing as mp

from dataset.cwru.sehri_et_al import get_list_of_papers_splits, get_list_of_X_y, load_matlab_acquisition


# -----------------------------
# RMS preprocessing (per-sample)
# -----------------------------
def rms_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    RMS normalization per sample/segment.
    X expected shape: (N, L, C) or (N, L) -> we handle both.

    For each sample i:
        X_i <- X_i / (rms(X_i) + eps)
    """
    if X.ndim == 2:
        # (N, L) -> treat as (N, L, 1)
        X_ = X[:, :, None]
    elif X.ndim == 3:
        X_ = X
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    rms = np.sqrt(np.mean(X_ ** 2, axis=(1, 2), keepdims=True))  # (N, 1, 1)
    Xn = X_ / (rms + eps)

    return Xn if X.ndim == 3 else Xn[:, :, 0]


# -----------------------------
# CNN Backbone (unchanged)
# -----------------------------
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
        return self.features(x)  # (B, 256, 1, T)


# -----------------------------
# Gradient Reversal Layer (GRL)
# -----------------------------
from torch.autograd import Function

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _GradReverse.apply(x, self.lambd)


# -----------------------------
# Domain head
# -----------------------------
class DomainHead(nn.Module):
    def __init__(self, in_features=256, hidden=128, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feat):
        return self.net(feat)


# -----------------------------
# CNN + LSTM + (class head + domain head)
# -----------------------------
class CNNLSTM_DANN(nn.Module):
    def __init__(self, num_classes, num_domains=2, grl_lambda=1.0):
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

        self.grl = GradientReversal(lambd=grl_lambda)
        self.domain_head = DomainHead(in_features=256, hidden=128, num_domains=num_domains)

    def extract_feat(self, x):
        x = self.cnn(x)          # (B, 256, 1, T)
        x = x.squeeze(2)         # (B, 256, T)
        x = x.permute(0, 2, 1)   # (B, T, 256)
        lstm_out, _ = self.lstm(x)
        feat = lstm_out[:, -1, :]  # (B, 256)
        return feat

    def forward(self, x):
        feat = self.extract_feat(x)
        class_logits = self.classifier(feat)

        rev_feat = self.grl(feat)             # gradients reversed here
        domain_logits = self.domain_head(rev_feat)

        return class_logits, domain_logits


# -----------------------------
# Dataset that provides: (x, y_fault, y_domain, is_labeled)
# -----------------------------
class DomainDataset(Dataset):
    def __init__(self, X, y_fault, domain_id: int, labeled: bool):
        self.X = X
        self.y_fault = y_fault  # torch.LongTensor or None
        self.domain_id = domain_id
        self.labeled = labeled

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        d = torch.tensor(self.domain_id, dtype=torch.long)

        if self.labeled:
            y = self.y_fault[idx]
            labeled = torch.tensor(True)
        else:
            y = torch.tensor(0, dtype=torch.long)  # dummy
            labeled = torch.tensor(False)

        return x, y, d, labeled


def collate_source(batch):
    # batch: list of (x, y, d, labeled)
    xb = torch.stack([b[0] for b in batch], dim=0)
    yb = torch.stack([b[1] for b in batch], dim=0)
    db = torch.stack([b[2] for b in batch], dim=0)
    lb = torch.stack([b[3] for b in batch], dim=0)
    return xb, yb, db, lb


# -----------------------------
# Training (DANN) - source-only (0.007 & 0.014)
# -----------------------------
def train_one_epoch_dann_source_only(
    model,
    source_loader,
    optimizer,
    criterion_fault,
    criterion_domain,
    device,
    domain_loss_weight=1.0
):
    model.train()
    total_loss = 0.0
    total_fault = 0.0
    total_domain = 0.0
    steps = 0

    for xb, yb, db, lb in source_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        db = db.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        class_logits, domain_logits = model(xb)

        # Domain loss on source domains (0.007 vs 0.014)
        loss_domain = criterion_domain(domain_logits, db)

        # Fault loss on labeled samples (all are labeled here)
        loss_fault = criterion_fault(class_logits[lb], yb[lb])

        loss = loss_fault + domain_loss_weight * loss_domain
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_fault += float(loss_fault.item())
        total_domain += float(loss_domain.item())
        steps += 1

    return total_loss / steps, total_fault / steps, total_domain / steps


@torch.no_grad()
def predict_all_class(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits, _ = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(yb.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    mp.freeze_support()

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

    # Train domains: 0.007 and 0.014 | Test domain: 0.021
    Xtr, ytr, Xtr2, ytr2, Xte, yte = (
        list_of_X_y[0][0], list_of_X_y[0][1],  # 0.007
        list_of_X_y[1][0], list_of_X_y[1][1],  # 0.014
        list_of_X_y[2][0], list_of_X_y[2][1],  # 0.021 (TEST only)
    )
    print(Xtr.shape, ytr.shape, " / ", Xtr2.shape, ytr2.shape, " / ", Xte.shape, yte.shape)

    # --- RMS preprocessing (IMPORTANT: apply to all splits) ---
    Xtr  = rms_normalize_np(Xtr)
    Xtr2 = rms_normalize_np(Xtr2)
    Xte  = rms_normalize_np(Xte)

    # --- Label encoding (fit on ALL labeled training labels: ytr + ytr2) ---
    y_train_all = np.concatenate([ytr, ytr2], axis=0)
    unique_label = np.unique(y_train_all)
    label_to_index = {label: idx for idx, label in enumerate(unique_label)}

    ytr_encoded  = np.array([label_to_index[label] for label in ytr], dtype=np.int64)
    ytr2_encoded = np.array([label_to_index[label] for label in ytr2], dtype=np.int64)

    unseen = set(np.unique(yte)) - set(unique_label)
    if len(unseen) > 0:
        raise ValueError(f"Test set has unseen labels not present in train: {unseen}")
    yte_encoded = np.array([label_to_index[label] for label in yte], dtype=np.int64)

    # --- To torch: (N, L, C) -> (N, C, 1, L) ---
    Xtr_torch  = torch.from_numpy(Xtr).float().permute(0, 2, 1).unsqueeze(2)
    Xtr2_torch = torch.from_numpy(Xtr2).float().permute(0, 2, 1).unsqueeze(2)
    Xte_torch  = torch.from_numpy(Xte).float().permute(0, 2, 1).unsqueeze(2)

    ytr_torch  = torch.from_numpy(ytr_encoded).long()
    ytr2_torch = torch.from_numpy(ytr2_encoded).long()
    yte_torch  = torch.from_numpy(yte_encoded).long()

    # --- Source domains only (train) ---
    # Domain ids: 0 -> severity 0.007, 1 -> severity 0.014
    ds_src_007 = DomainDataset(Xtr_torch,  ytr_torch,  domain_id=0, labeled=True)
    ds_src_014 = DomainDataset(Xtr2_torch, ytr2_torch, domain_id=1, labeled=True)
    ds_source = torch.utils.data.ConcatDataset([ds_src_007, ds_src_014])

    # --- Train/Val split from the source only ---
    N = len(ds_source)
    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    tr_idx = idx[:split]
    va_idx = idx[split:]

    source_train = Subset(ds_source, tr_idx)
    source_val   = Subset(ds_source, va_idx)

    source_train_loader = DataLoader(
        source_train, batch_size=64, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=2, collate_fn=collate_source
    )
    source_val_loader = DataLoader(
        source_val, batch_size=64, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=2, collate_fn=collate_source
    )

    # Test loader (standard: x,y) - domain 0.021 only
    test_ds = torch.utils.data.TensorDataset(Xte_torch, yte_torch)
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=2
    )

    # --- Model ---
    num_classes = len(unique_label)
    model = CNNLSTM_DANN(num_classes=num_classes, num_domains=2, grl_lambda=1.0).to(device)

    criterion_fault = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 60
    domain_loss_weight = 0.5  # tune this

    for epoch in range(1, epochs + 1):
        loss_total, loss_f, loss_d = train_one_epoch_dann_source_only(
            model,
            source_train_loader,
            optimizer,
            criterion_fault,
            criterion_domain,
            device,
            domain_loss_weight=domain_loss_weight
        )
        print(f"Epoch {epoch:02d} | total={loss_total:.4f} | fault={loss_f:.4f} | domain={loss_d:.4f}")

    # --- Evaluate on test domain (0.021) ---
    y_true_te, y_pred_te = predict_all_class(model, test_loader, device)
    cm_te = confusion_matrix(y_true_te, y_pred_te)
    print("Test confusion matrix:\n", cm_te)

    index_to_label = {v: k for k, v in label_to_index.items()}
    target_names = [index_to_label[i] for i in range(len(index_to_label))]
    print("\nTest classification report:\n")
    print(classification_report(y_true_te, y_pred_te, target_names=target_names, digits=4))
