import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fftpack import dct
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


JOINTS = list(range(15)) + [19, 21, 22, 24]
MAP_LABEL = {
    "SQUAT": [1, 2, 3, 4, 5, 10],
    "Lunges": [1, 4, 6],
    "Plank": [1, 7, 8],
}
OFFSET = {"SQUAT": 0, "Lunges": 6, "Plank": 9}
CORRECT_CLASSES = np.array([0, 6, 9])
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_ec3d_sequences(data_path):
    with Path(data_path).open("rb") as f:
        raw = pickle.load(f)

    labels = pd.DataFrame(raw["labels"], columns=["act", "sub", "lab", "rep", "frame"])
    labels[["lab", "rep"]] = labels[["lab", "rep"]].astype(int)
    poses = raw["poses"][:, :, JOINTS]

    xs, ys, subjects, meta = [], [], [], []
    for key, idx in labels.groupby(["act", "sub", "lab", "rep"], sort=False).groups.items():
        act, subject, label, rep = key
        sequence = poses[list(idx)].reshape(len(idx), -1).T
        coeff = dct(sequence, axis=1, norm="ortho")
        if coeff.shape[1] < 25:
            coeff = np.hstack([coeff, np.repeat(coeff[:, -1:], 25 - coeff.shape[1], axis=1)])

        xs.append(coeff[:, :25].astype("float32"))
        ys.append(OFFSET[act] + MAP_LABEL[act].index(label))
        subjects.append(subject)
        meta.append({"act": act, "subject": subject, "label": int(label), "rep": int(rep)})

    return np.stack(xs), np.array(ys), np.array(subjects), meta


class GraphConv(torch.nn.Module):
    def __init__(self, in_features, out_features, nodes=57):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.adj = torch.nn.Parameter(torch.empty(nodes, nodes))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.adj)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.matmul(self.adj, torch.matmul(x, self.weight)) + self.bias


class GCNClassifier(torch.nn.Module):
    def __init__(self, hidden=32, dropout=0.4, classes=12):
        super().__init__()
        self.g1 = GraphConv(25, hidden)
        self.bn1 = torch.nn.BatchNorm1d(57 * hidden)
        self.g2 = GraphConv(hidden, 25)
        self.bn2 = torch.nn.BatchNorm1d(57 * 25)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(57 * 25, classes)

    def forward(self, x):
        batch = x.shape[0]
        x = self.g1(x)
        x = self.bn1(x.reshape(batch, -1)).reshape(batch, 57, -1)
        x = self.dropout(torch.relu(x))
        x = self.g2(x)
        x = self.bn2(x.reshape(batch, -1)).reshape(batch, 57, 25)
        x = self.dropout(torch.relu(x))
        return self.fc(x.reshape(batch, -1))


def class_weights(y):
    counts = np.bincount(y, minlength=12).astype("float32")
    weights = counts.sum() / np.maximum(counts, 1)
    return (weights / weights.mean()).astype("float32")


def train_once(x, y, train_idx, val_idx, device, hidden, dropout, lr, epochs=250, patience=35):
    xt = torch.tensor(x[train_idx], device=device)
    yt = torch.tensor(y[train_idx], device=device, dtype=torch.long)
    xv = torch.tensor(x[val_idx], device=device)
    yv = y[val_idx]

    model = GCNClassifier(hidden=hidden, dropout=dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(y[train_idx]), device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), 64):
            batch = order[start : start + 64]
            loss = loss_fn(model(xt[batch]), yt[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).cpu().numpy()
        score = f1_score(yv, pred, average="macro")
        if score > best_score + 1e-6:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_score, best_epoch


def train_fixed_epochs(x, y, train_idx, device, hidden, dropout, lr, epochs):
    xt = torch.tensor(x[train_idx], device=device)
    yt = torch.tensor(y[train_idx], device=device, dtype=torch.long)
    model = GCNClassifier(hidden=hidden, dropout=dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(y[train_idx]), device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), 64):
            batch = order[start : start + 64]
            loss = loss_fn(model(xt[batch]), yt[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def metrics(y_true, y_pred):
    binary_true = np.isin(y_true, CORRECT_CLASSES).astype(int)
    binary_pred = np.isin(y_pred, CORRECT_CLASSES).astype(int)
    return {
        "multi_accuracy": float(accuracy_score(y_true, y_pred)),
        "multi_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "multi_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "binary_accuracy": float(accuracy_score(binary_true, binary_pred)),
        "binary_balanced_accuracy": float(balanced_accuracy_score(binary_true, binary_pred)),
        "binary_f1": float(f1_score(binary_true, binary_pred)),
        "multi_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "binary_confusion_matrix": confusion_matrix(binary_true, binary_pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "external/ec3d_repo/data/EC3D/data_3D.pickle"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "codex/artifacts"))
    parser.add_argument("--test-subject", default="Vidit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ponytail: EC3D DCT sequence model; add REHAB/video only after this ceiling is measured.
    x, y, subjects, meta = build_ec3d_sequences(args.data)
    train_subjects = [s for s in np.unique(subjects) if s != args.test_subject]
    test_idx = np.where(subjects == args.test_subject)[0]

    candidates = []
    for hidden in [16, 32, 64, 128]:
        for dropout in [0.2, 0.4, 0.6]:
            for lr in [0.001, 0.003]:
                scores, epochs = [], []
                for val_subject in train_subjects:
                    train_idx = np.where((subjects != args.test_subject) & (subjects != val_subject))[0]
                    val_idx = np.where(subjects == val_subject)[0]
                    _, score, epoch = train_once(
                        x, y, train_idx, val_idx, device, hidden, dropout, lr
                    )
                    scores.append(score)
                    epochs.append(epoch)
                candidates.append(
                    {
                        "hidden": hidden,
                        "dropout": dropout,
                        "lr": lr,
                        "mean_val_macro_f1": float(np.mean(scores)),
                        "fold_val_macro_f1": scores,
                        "fold_best_epochs": epochs,
                        "final_epochs": int(np.median(epochs)),
                    }
                )

    best = max(candidates, key=lambda c: c["mean_val_macro_f1"])
    final_train_idx = np.where(subjects != args.test_subject)[0]
    model = train_fixed_epochs(
        x,
        y,
        final_train_idx,
        device,
        best["hidden"],
        best["dropout"],
        best["lr"],
        best["final_epochs"],
    )

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(x[final_train_idx], device=device)).argmax(1).cpu().numpy()
        test_pred = model(torch.tensor(x[test_idx], device=device)).argmax(1).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_params": best,
            "class_map": {"correct_classes": CORRECT_CLASSES.tolist(), "offset": OFFSET, "label_map": MAP_LABEL},
        },
        out_dir / "ec3d_gcn.pt",
    )

    report = {
        "device": str(device),
        "data_path": args.data,
        "test_subject": args.test_subject,
        "n_sequences": int(len(y)),
        "train_subjects": train_subjects,
        "test_sequences": int(len(test_idx)),
        "best_params": best,
        "test": metrics(y[test_idx], test_pred),
        "train": metrics(y[final_train_idx], train_pred),
        "candidates": sorted(candidates, key=lambda c: c["mean_val_macro_f1"], reverse=True),
    }
    (out_dir / "ec3d_gcn_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["device", "best_params", "test", "train"]}, indent=2))


if __name__ == "__main__":
    main()
