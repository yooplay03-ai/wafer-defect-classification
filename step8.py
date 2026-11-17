import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

KEY = "X_max"
EPOCHS = 25

# 1. 데이터 불러오기와 로트 단위 나누기
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

# 2. 한 장씩 따로 증강하기
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

# 3. CNN
def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
        nn.Linear(64, len(classes)),
    )

# 4. 학습하고 평가하기
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

# 5. 시드 3개로 실행
mlflow.set_experiment("wafer-defect")
macros, per = [], []
for seed in [0, 1, 2]:
    macro, per_class = train_eval(seed)
    macros.append(macro)
    per.append(per_class)
    print(f"aug_each seed {seed}: macro F1 {macro:.3f}")
    with mlflow.start_run(run_name=f"cnn_aug_each_25ep_seed{seed}"):
        mlflow.log_param("resize", KEY)
        mlflow.log_param("augment", "rot90 + flip, per sample")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("seed", seed)
        mlflow.log_metric("macro_f1", macro)
        for c, f in zip(classes, per_class):
            mlflow.log_metric(f"f1_{c}", f)

# 6. 이전 결과와 비교 (step7, 25 epoch)
plain = [0.948, 0.814, 0.880, 0.981, 0.783, 0.752, 0.794, 0.562]
aug_batch = [0.952, 0.821, 0.878, 0.981, 0.804, 0.704, 0.794, 0.695]
aug_each = np.mean(per, axis=0)

print("\n[25 epoch macro F1]")
print("plain:      0.814 ± 0.004")
print("aug_batch:  0.829 ± 0.020")
print(f"aug_each:   {np.mean(macros):.3f} ± {np.std(macros):.3f}")

print("\n패턴별 F1 (plain / aug_batch / aug_each)")
for c, a, b, e in zip(classes, plain, aug_batch, aug_each):
    print(f"{c:10s} {a:.3f} / {b:.3f} / {e:.3f}")