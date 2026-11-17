# 5-fold 최종 평가
import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

KEY = "X_max"
EPOCHS = 25

d = np.load("data/wafer64.npz")
X, y, lot = d[KEY], d["y"], d["lot"]
classes = list(d["classes"])

def to_tensor(a):
    onehot = np.eye(3, dtype=np.float32)[a]
    return torch.from_numpy(onehot).permute(0, 3, 1, 2)

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

def train_predict(X_tr, y_tr, X_te, seed=0, name=""):
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
            print(f"  {name}: {epoch + 1}/{EPOCHS} epoch")
    model.eval()
    with torch.no_grad():
        return model(X_te).argmax(dim=1).numpy()

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
mlflow.set_experiment("wafer-defect")
all_true, all_pred, macros = [], [], []

for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=lot)):
    pred = train_predict(to_tensor(X[tr]), torch.tensor(y[tr]), to_tensor(X[te]), name=f"fold {fold}")
    macro = f1_score(y[te], pred, average="macro")
    macros.append(macro)
    all_true.append(y[te])
    all_pred.append(pred)
    print(f"fold {fold}: macro F1 {macro:.3f}")
    with mlflow.start_run(run_name=f"final_5fold_fold{fold}"):
        mlflow.log_param("model", "CNN 4층 + aug_each, 25ep, 64x64 max")
        mlflow.log_param("fold", fold)
        mlflow.log_metric("macro_f1", macro)

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)
print(f"\n5-fold macro F1: {np.mean(macros):.3f} ± {np.std(macros):.3f}")
print(classification_report(all_true, all_pred, target_names=classes, digits=3))
print(confusion_matrix(all_true, all_pred))