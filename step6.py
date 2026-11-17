import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

KEY = "X_max"

d = np.load("data/wafer32.npz", allow_pickle=True)
X, y, lot = d[KEY], d["y"], d["lot"]
classes = list(d["classes"])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = next(sgkf.split(X, y, groups=lot))

def to_tensor(a):
    onehot = np.eye(3, dtype=np.float32)[a]
    return torch.from_numpy(onehot).permute(0, 3, 1, 2)

X_tr, X_te = to_tensor(X[tr]), to_tensor(X[te])
y_tr = torch.tensor(y[tr])

# 패턴별 가중치: 적은 패턴일수록 크게
counts = np.bincount(y[tr], minlength=len(classes))
weights = len(tr) / (len(classes) * counts)
for c, n, w in zip(classes, counts, weights):
    print(f"{c:10s} 개수 {n:5d}  가중치 {w:.2f}")

def augment(xb):
    k = int(torch.randint(0, 4, (1,)))
    xb = torch.rot90(xb, k, dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=(3,))
    return xb

def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
        nn.Linear(64, len(classes)),
    )

def train_eval(use_aug, seed):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for xb, yb in loader:
            if use_aug:
                xb = augment(xb)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(dim=1).numpy()
    macro = f1_score(y[te], pred, average="macro")
    per_class = f1_score(y[te], pred, average=None)
    return macro, per_class

mlflow.set_experiment("wafer-defect")
macros, per = [], []
for seed in [0, 1, 2]:
    macro, per_class = train_eval(True, seed)
    macros.append(macro)
    per.append(per_class)
    print(f"aug seed {seed}: macro F1 {macro:.3f}")
    with mlflow.start_run(run_name=f"cnn_aug_seed{seed}"):
        mlflow.log_param("resize", KEY)
        mlflow.log_param("augment", "rot90 + flip")
        mlflow.log_param("seed", seed)
        mlflow.log_metric("macro_f1", macro)
        for c, f in zip(classes, per_class):
            mlflow.log_metric(f"f1_{c}", f)

plain = [0.953, 0.811, 0.874, 0.981, 0.771, 0.745, 0.779, 0.459]   # step4 결과
aug = np.mean(per, axis=0)
print(f"\nplain: macro F1 0.797 ± 0.005")
print(f"aug:   macro F1 {np.mean(macros):.3f} ± {np.std(macros):.3f}")
print("\n패턴별 평균 F1 (plain → aug)")
for c, a, b in zip(classes, plain, aug):
    print(f"{c:10s} {a:.3f} → {b:.3f}")