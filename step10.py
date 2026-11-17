#64X64 층 추가하기

import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

EPOCHS = 25

d = np.load("data/wafer64.npz")
X, y, lot = d["X_max"], d["y"], d["lot"]
classes = list(d["classes"])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = next(sgkf.split(X, y, groups=lot))

def to_tensor(a):
    onehot = np.eye(3, dtype=np.float32)[a]
    return torch.from_numpy(onehot).permute(0, 3, 1, 2)

X_tr, X_te = to_tensor(X[tr]), to_tensor(X[te])
y_tr = torch.tensor(y[tr])

def augment_each(xb):
    out = xb.clone()
    ks = torch.randint(0, 4, (len(xb),))
    for k in range(1, 4):
        idx = ks == k
        if idx.any():
            out[idx] = torch.rot90(xb[idx], k, dims=(2, 3))
    flip = torch.rand(len(xb)) < 0.5
    out[flip] = torch.flip(out[flip], dims=(3,))
    return out

def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
        nn.Linear(64, len(classes)),
    )

def train_eval(seed):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb = augment_each(xb)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  seed {seed}: {epoch + 1}/{EPOCHS} epoch")
    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(dim=1).numpy()
    return f1_score(y[te], pred, average="macro"), f1_score(y[te], pred, average=None)

mlflow.set_experiment("wafer-defect")
macros, per = [], []
for seed in [0, 1, 2]:
    macro, per_class = train_eval(seed)
    macros.append(macro)
    per.append(per_class)
    print(f"64x64 seed {seed}: macro F1 {macro:.3f}")
    with mlflow.start_run(run_name=f"cnn_64_aug_each_seed{seed}"):
        mlflow.log_param("resize", "X_max 64x64")
        mlflow.log_param("augment", "rot90 + flip, per sample")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("seed", seed)
        mlflow.log_metric("macro_f1", macro)
        for c, f in zip(classes, per_class):
            mlflow.log_metric(f"f1_{c}", f)

aug32 = [0.949, 0.795, 0.883, 0.983, 0.794, 0.755, 0.795, 0.671]
res64 = np.mean(per, axis=0)
print("\n[aug_each, 25 epoch]")
print("32x32: 0.828 ± 0.005")
print(f"64x64: {np.mean(macros):.3f} ± {np.std(macros):.3f}")
print("\n패턴별 F1 (32x32 → 64x64)")
for c, a, b in zip(classes, aug32, res64):
    print(f"{c:10s} {a:.3f} → {b:.3f}")